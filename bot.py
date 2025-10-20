import os
import re
import asyncio
import logging
import aiohttp
import sqlite3
from dotenv import load_dotenv

from aiohttp import web
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove

load_dotenv()

TG_TOKEN = os.getenv("TG_TOKEN")
AI_API_KEY = os.getenv("AI_API_KEY")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

storage = MemoryStorage()
bot = Bot(token=TG_TOKEN)
dp = Dispatcher(storage=storage)

AI_API_URL = "https://api.intelligence.io.solutions/api/v1/chat/completions"
AI_MODEL = "deepseek-ai/DeepSeek-R1-0528"

DB_FILE = "kwork_bot.db"

# Константы для валидации
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 100
MIN_STACK_LENGTH = 10
MIN_VACANCY_LENGTH = 20


class UserStates(StatesGroup):
    waiting_for_name = State()
    waiting_for_gender = State()
    waiting_for_stack = State()
    ready_for_vacancy = State()
    choosing_update = State()


def init_db():
    """Инициализация базы данных с обработкой ошибок"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_stacks'")
        table_exists = cur.fetchone()

        if table_exists:
            cur.execute("PRAGMA table_info(user_stacks)")
            columns = [column[1] for column in cur.fetchall()]

            if 'name' not in columns:
                cur.execute("ALTER TABLE user_stacks ADD COLUMN name TEXT")
            if 'gender' not in columns:
                cur.execute("ALTER TABLE user_stacks ADD COLUMN gender TEXT")
        else:
            cur.execute('''
                CREATE TABLE user_stacks (
                    user_id INTEGER PRIMARY KEY,
                    name TEXT,
                    gender TEXT,
                    tech_stack TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        conn.commit()
        conn.close()
        logger.info("✅ База данных успешно инициализирована")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации БД: {e}")
        raise


def save_user_stack(user_id: int, stack: str, name: str = None, gender: str = None):
    """Сохранение данных пользователя с обработкой ошибок"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO user_stacks (user_id, tech_stack, name, gender)
            VALUES (?, ?, ?, ?)
        ''', (user_id, stack, name, gender))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения данных пользователя {user_id}: {e}")
        raise


def update_user_info(user_id: int, name: str = None, gender: str = None):
    """Обновление информации пользователя с обработкой ошибок"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()

        cur.execute(
            'SELECT name, gender, tech_stack FROM user_stacks WHERE user_id = ?', (user_id,))
        result = cur.fetchone()

        if result:
            current_name, current_gender, current_stack = result
            new_name = name if name is not None else current_name
            new_gender = gender if gender is not None else current_gender

            cur.execute('''
                UPDATE user_stacks 
                SET name = ?, gender = ?
                WHERE user_id = ?
            ''', (new_name, new_gender, user_id))
        else:
            cur.execute('''
                INSERT INTO user_stacks (user_id, name, gender, tech_stack)
                VALUES (?, ?, ?, ?)
            ''', (user_id, name, gender, ''))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Ошибка обновления данных пользователя {user_id}: {e}")
        raise


def get_user_data(user_id: int):
    """Получение данных пользователя с обработкой ошибок"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(
            'SELECT name, gender, tech_stack FROM user_stacks WHERE user_id = ?', (user_id,))
        result = cur.fetchone()
        conn.close()
        if result:
            return {
                'name': result[0],
                'gender': result[1],
                'tech_stack': result[2]
            }
        return None
    except Exception as e:
        logger.error(f"❌ Ошибка получения данных пользователя {user_id}: {e}")
        return None


def is_profile_complete(user_id: int):
    """Проверка полноты профиля"""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return False
        return all([
            user_data.get('name'),
            user_data.get('gender'),
            user_data.get('tech_stack')
        ])
    except Exception as e:
        logger.error(f"❌ Ошибка проверки профиля {user_id}: {e}")
        return False


def get_update_keyboard():
    """Клавиатура для выбора обновления"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Имя"), KeyboardButton(text="Пол")],
            [KeyboardButton(text="Стек"), KeyboardButton(text="Все вместе")]
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    return keyboard


def get_gender_keyboard():
    """Клавиатура для выбора пола"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Мужской"), KeyboardButton(text="Женский")]
        ],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    return keyboard


def validate_name(name: str) -> tuple[bool, str]:
    """Валидация имени"""
    if len(name) < MIN_NAME_LENGTH:
        return False, f"❌ Имя слишком короткое. Минимум {MIN_NAME_LENGTH} символа."
    if len(name) > MAX_NAME_LENGTH:
        return False, f"❌ Имя слишком длинное. Максимум {MAX_NAME_LENGTH} символов."
    if not re.match(r'^[а-яА-ЯёЁa-zA-Z\s\-]+$', name):
        return False, "❌ Имя может содержать только буквы, пробелы и дефисы."
    return True, ""


async def ask_ai_api(system_prompt: str, user_prompt: str, max_retries: int = 3):
    """Запрос к AI API с обработкой ошибок и повторными попытками"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_API_KEY}"
    }

    data = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(AI_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=90)) as response:
                    response_text = await response.text()
                    logger.info(
                        f"API response (attempt {attempt + 1}): {response_text[:200]}...")

                    if response.status == 200:
                        try:
                            result = await response.json()
                            text = result['choices'][0]['message']['content']

                            cleaned_text = re.sub(
                                r'<think>.*?</think>', '', text, flags=re.DOTALL)
                            cleaned_text = cleaned_text.strip()

                            return cleaned_text
                        except (KeyError, IndexError) as e:
                            logger.error(f"❌ Ошибка парсинга JSON: {e}")
                            if attempt == max_retries - 1:
                                return "Извините, не удалось обработать ответ AI. Попробуйте позже."
                    else:
                        logger.error(f"❌ API вернул статус {response.status}")
                        if attempt == max_retries - 1:
                            return f"Извините, API временно недоступен (код {response.status}). Попробуйте позже."

        except asyncio.TimeoutError:
            logger.error(f"❌ Таймаут API (попытка {attempt + 1})")
            if attempt == max_retries - 1:
                return "Извините, превышено время ожидания ответа. Попробуйте позже."
        except aiohttp.ClientError as e:
            logger.error(f"❌ Ошибка соединения с API: {e}")
            if attempt == max_retries - 1:
                return "Извините, ошибка соединения с сервером. Попробуйте позже."
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка API: {e}")
            if attempt == max_retries - 1:
                return "Извините, произошла неожиданная ошибка. Попробуйте позже."

        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return "Извините, не удалось получить ответ после нескольких попыток."


@dp.message(Command('start'))
async def cmd_start(message: types.Message, state: FSMContext):
    """Команда /start с обработкой ошибок"""
    try:
        user_id = message.from_user.id
        logger.info(f"Пользователь {user_id} запустил /start")

        if is_profile_complete(user_id):
            user_data = get_user_data(user_id)
            info_text = f"👤 Имя: {user_data['name']}\n"
            info_text += f"⚧ Пол: {user_data['gender']}\n"
            info_text += f"💼 Стек: <code>{user_data['tech_stack']}</code>\n"

            await message.answer(
                f"👋 С возвращением!\n\n"
                f"Ваши данные:\n{info_text}\n"
                f"Отправьте текст вакансии, и я создам отклик.\n"
                f"Или используйте /update чтобы обновить данные.",
                parse_mode="HTML",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.ready_for_vacancy)
        else:
            await message.answer(
                "👋 Привет! Я бот для создания откликов на вакансии Kwork.\n\n"
                "Для начала работы мне нужно узнать о вас.\n\n"
                "Как вас зовут? (От 2 до 100 символов)",
                parse_mode="HTML",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.waiting_for_name)
    except Exception as e:
        logger.error(f"❌ Ошибка в cmd_start: {e}")
        await message.answer("Произошла ошибка. Попробуйте еще раз.")


@dp.message(Command('update'))
async def cmd_update(message: types.Message, state: FSMContext):
    """Команда /update с обработкой ошибок"""
    try:
        user_id = message.from_user.id

        if not is_profile_complete(user_id):
            await message.answer(
                "❌ Сначала заполните все данные профиля!\n"
                "Используйте /start для заполнения.",
                reply_markup=ReplyKeyboardRemove()
            )
            return

        await message.answer(
            "Что вы хотите обновить?\n\n"
            "Выберите из меню ниже:",
            reply_markup=get_update_keyboard()
        )
        await state.set_state(UserStates.choosing_update)
    except Exception as e:
        logger.error(f"❌ Ошибка в cmd_update: {e}")
        await message.answer("Произошла ошибка. Попробуйте еще раз.")


@dp.message(UserStates.choosing_update)
async def handle_update_choice(message: types.Message, state: FSMContext):
    """Обработка выбора обновления"""
    try:
        choice = message.text.strip().lower()

        if choice == "имя":
            await message.answer(
                "Введите ваше новое имя (от 2 до 100 символов):",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.waiting_for_name)
        elif choice == "пол":
            await message.answer(
                "Укажите ваш пол:",
                reply_markup=get_gender_keyboard()
            )
            await state.set_state(UserStates.waiting_for_gender)
        elif choice == "стек":
            await message.answer(
                "📝 Отправьте обновленный стек технологий и опыт:\n\n"
                "Например:\n"
                "<code>HTML, CSS, JavaScript, TypeScript, React, Node.js\n"
                "3 года в веб-разработке</code>",
                parse_mode="HTML",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.waiting_for_stack)
        elif choice == "все вместе":
            await message.answer(
                "Хорошо, давайте обновим все данные.\n\n"
                "Как вас зовут? (От 2 до 100 символов)",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.update_data(updating_all=True)
            await state.set_state(UserStates.waiting_for_name)
        else:
            await message.answer(
                "❌ Пожалуйста, выберите кнопку из меню ниже:",
                reply_markup=get_update_keyboard()
            )
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_update_choice: {e}")
        await message.answer("Произошла ошибка. Попробуйте еще раз.")


@dp.message(Command('mystack'))
async def cmd_mystack(message: types.Message):
    """Команда /mystack с обработкой ошибок"""
    try:
        user_id = message.from_user.id
        user_data = get_user_data(user_id)

        if user_data:
            info_text = ""
            if user_data['name']:
                info_text += f"👤 Имя: {user_data['name']}\n"
            if user_data['gender']:
                info_text += f"⚧ Пол: {user_data['gender']}\n"
            if user_data['tech_stack']:
                info_text += f"💼 Стек: <code>{user_data['tech_stack']}</code>\n"

            await message.answer(
                f"Ваши текущие данные:\n\n{info_text}\n"
                f"Используйте /update для изменения.",
                parse_mode="HTML"
            )
        else:
            await message.answer(
                "У вас еще не сохранены данные.\n"
                "Используйте /start для начала работы."
            )
    except Exception as e:
        logger.error(f"❌ Ошибка в cmd_mystack: {e}")
        await message.answer("Произошла ошибка. Попробуйте еще раз.")


@dp.message(UserStates.waiting_for_name)
async def handle_name_input(message: types.Message, state: FSMContext):
    """Обработка ввода имени с валидацией"""
    try:
        user_id = message.from_user.id
        name = message.text.strip()

        is_valid, error_msg = validate_name(name)
        if not is_valid:
            await message.answer(error_msg)
            return

        update_user_info(user_id, name=name)

        data = await state.get_data()
        updating_all = data.get('updating_all', False)
        is_complete = is_profile_complete(user_id)

        if updating_all or not is_complete:
            await message.answer(
                f"✅ Имя сохранено: {name}\n\n"
                "Теперь укажите ваш пол:",
                reply_markup=get_gender_keyboard()
            )
            await state.set_state(UserStates.waiting_for_gender)
        else:
            await message.answer(
                f"✅ Имя успешно изменено: {name}\n\n"
                "Отправьте текст вакансии для создания отклика.",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.ready_for_vacancy)
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_name_input: {e}")
        await message.answer("Произошла ошибка при сохранении имени. Попробуйте еще раз.")


@dp.message(UserStates.waiting_for_gender)
async def handle_gender_input(message: types.Message, state: FSMContext):
    """Обработка выбора пола"""
    try:
        user_id = message.from_user.id
        gender = message.text.strip().lower()

        if gender not in ['мужской', 'женский']:
            await message.answer(
                "❌ Пожалуйста, выберите 'Мужской' или 'Женский' из кнопок.",
                reply_markup=get_gender_keyboard()
            )
            return

        update_user_info(user_id, gender=gender)

        data = await state.get_data()
        updating_all = data.get('updating_all', False)

        user_data = get_user_data(user_id)
        was_complete = all(
            [user_data.get('name'), user_data.get('tech_stack')])

        if updating_all or not was_complete:
            await message.answer(
                f"✅ Пол сохранен: {gender}\n\n"
                f"Теперь отправьте ваш стек технологий и опыт разработки (минимум {MIN_STACK_LENGTH} символов).\n\n"
                "Например:\n"
                "<code>HTML, CSS, JavaScript, TypeScript, React, Node.js\n"
                "3 года в веб-разработке</code>",
                parse_mode="HTML",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.waiting_for_stack)
        else:
            await message.answer(
                f"✅ Пол успешно изменен: {gender}\n\n"
                "Отправьте текст вакансии для создания отклика.",
                reply_markup=ReplyKeyboardRemove()
            )
            await state.set_state(UserStates.ready_for_vacancy)
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_gender_input: {e}")
        await message.answer("Произошла ошибка при сохранении пола. Попробуйте еще раз.")


@dp.message(UserStates.waiting_for_stack)
async def handle_stack_input(message: types.Message, state: FSMContext):
    """Обработка ввода стека технологий"""
    try:
        user_id = message.from_user.id
        stack = message.text.strip()

        if len(stack) < MIN_STACK_LENGTH:
            await message.answer(
                f"❌ Слишком короткий стек. Минимум {MIN_STACK_LENGTH} символов. "
                f"Опишите подробнее ваши навыки и опыт."
            )
            return

        user_data = get_user_data(user_id)
        if user_data:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute(
                'UPDATE user_stacks SET tech_stack = ? WHERE user_id = ?', (stack, user_id))
            conn.commit()
            conn.close()
        else:
            save_user_stack(user_id, stack)

        data = await state.get_data()
        updating_all = data.get('updating_all', False)

        if updating_all:
            await state.update_data(updating_all=False)
            await message.answer(
                "✅ Все данные успешно обновлены!\n\n"
                f"<code>{stack}</code>\n\n"
                "Теперь отправьте текст вакансии, и я создам для вас отклик!",
                parse_mode="HTML",
                reply_markup=ReplyKeyboardRemove()
            )
        else:
            if is_profile_complete(user_id):
                was_first_setup = not user_data or not user_data.get(
                    'tech_stack')
                if was_first_setup:
                    await message.answer(
                        "✅ Отлично! Все данные заполнены!\n\n"
                        f"<code>{stack}</code>\n\n"
                        "Теперь отправьте текст вакансии, и я создам для вас отклик!",
                        parse_mode="HTML",
                        reply_markup=ReplyKeyboardRemove()
                    )
                else:
                    await message.answer(
                        "✅ Стек успешно изменен!\n\n"
                        f"<code>{stack}</code>\n\n"
                        "Отправьте текст вакансии для создания отклика.",
                        parse_mode="HTML",
                        reply_markup=ReplyKeyboardRemove()
                    )
            else:
                await message.answer(
                    "⚠️ Заполните остальные данные профиля через /start",
                    reply_markup=ReplyKeyboardRemove()
                )
                return

        await state.set_state(UserStates.ready_for_vacancy)
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_stack_input: {e}")
        await message.answer("Произошла ошибка при сохранении стека. Попробуйте еще раз.")


@dp.message(UserStates.ready_for_vacancy)
async def handle_vacancy(message: types.Message, state: FSMContext):
    """Обработка текста вакансии и создание отклика"""
    try:
        user_id = message.from_user.id
        vacancy_text = message.text.strip()

        if not is_profile_complete(user_id):
            await message.answer(
                "❌ Заполните все данные профиля!\n"
                "Используйте /start для заполнения.",
                reply_markup=ReplyKeyboardRemove()
            )
            return

        user_data = get_user_data(user_id)

        if len(vacancy_text) < MIN_VACANCY_LENGTH:
            await message.answer(
                f"❌ Текст вакансии слишком короткий (минимум {MIN_VACANCY_LENGTH} символов). "
                f"Отправьте полное описание вакансии."
            )
            return

        await message.answer("⏳ Создаю отклик, подождите...")

        performer_info = []
        performer_info.append(f"Имя: {user_data['name']}")
        performer_info.append(f"Пол: {user_data['gender']}")
        performer_info.append(f"Стек технологий: {user_data['tech_stack']}")

        performer_text = "\n".join(performer_info)

        system_prompt = (
            "Ты - бот, который создает ТОЛЬКО текст для отклика на вакансию и название для отклика до 100 символов. "
            "Ты не делаешь НИЧЕГО БОЛЕЕ, чем это. "
            "Ты ОБЯЗАН ответить в следующем формате:\n\n"
            "НАЗВАНИЕ: [название отклика до 100 символов]\n\n"
            "ТЕКСТ ОТКЛИКА:\n[текст отклика]\n\n"
            f"Информация об исполнителе:\n{performer_text}\n\n"
            "ВАЖНЫЕ ПРАВИЛА ДЛЯ ТЕКСТА ОТКЛИКА:\n"
            "1. НЕ выдумывай опыт и прошлые проекты! Не пиши про 'недавно делал', 'работал с похожими задачами', 'оптимизировал системы' и т.п.\n"
            "2. Пиши ТОЛЬКО о реальных навыках из стека исполнителя\n"
            "3. Используй простой, понятный язык - пиши так, чтобы понял обычный заказчик, не только программист\n"
            "4. Технологии упоминай кратко и по делу, без лишних технических терминов\n"
            "5. Фокусируйся на том, ЧТО ты можешь сделать для заказчика СЕЙЧАС, а не на прошлом опыте\n"
            "6. Будь конкретным и уверенным, покажи что понимаешь задачу\n"
            "7. Текст должен быть коротким, дружелюбным и профессиональным одновременно\n"
            "8. Не используй шаблонные фразы типа 'ваша вакансия мне интересна'\n"
            "9. Не делай жирного текста и других HTML-тегов, только переносы строк\n"
            "10. Не используй длинные тире (—), только короткие дефисы (-)\n\n"
            "Структура отклика:\n"
            "- Короткое приветствие\n"
            "- Покажи что понял задачу (1 предложение)\n"
            "- Какие навыки/технологии помогут решить задачу (2-4 пункта, кратко)\n"
            "- Что конкретно готов сделать\n"
            "- Призыв к действию (обсудить детали)\n\n"
            "Пример хорошего стиля: 'Готов помочь с вашим парсером. Работаю с Python и знаю как обходить защиты Google.'\n"
            "Пример плохого стиля: 'Ваша вакансия мне близка, недавно делал похожий проект с интеграцией AI для прогнозирования паттернов.'"
        )

        user_prompt = f"Вот текст вакансии:\n\n{vacancy_text}"

        response = await ask_ai_api(system_prompt, user_prompt)

        if len(response) > 4000:
            response = response[:4000] + '\n\n(ответ укорочен)'

        # Парсим ответ для разделения названия и текста отклика
        title_match = re.search(
            r'НАЗВАНИЕ:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        text_match = re.search(r'ТЕКСТ ОТКЛИКА:\s*(.+)',
                               response, re.IGNORECASE | re.DOTALL)

        if title_match and text_match:
            title = title_match.group(1).strip()
            text = text_match.group(1).strip()

            # Отправляем название без тегов - просто текст с разделителем
            await message.answer(
                "✅ Отклик готов!\n\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "📌 НАЗВАНИЕ:\n"
                f"{title}\n"
                "━━━━━━━━━━━━━━━━━━"
            )

            # Небольшая задержка между сообщениями
            await asyncio.sleep(0.3)

            # Отправляем текст отклика без тегов - просто текст с разделителем
            await message.answer(
                "━━━━━━━━━━━━━━━━━━\n"
                "📝 ТЕКСТ ОТКЛИКА:\n\n"
                f"{text}\n"
                "━━━━━━━━━━━━━━━━━━\n\n"
                "💡 Выделите текст для копирования"
            )
        else:
            # Если не удалось распарсить, отправляем как есть
            await message.answer(response)

        logger.info(f"✅ Отклик успешно создан для пользователя {user_id}")

    except Exception as e:
        logger.error(f"❌ Ошибка в handle_vacancy: {e}")
        await message.answer(
            "Произошла ошибка при создании отклика. Попробуйте еще раз или обратитесь к администратору."
        )


@dp.message()
async def handle_other(message: types.Message, state: FSMContext):
    """Обработка всех остальных сообщений"""
    try:
        current_state = await state.get_state()

        if current_state is None:
            await message.answer(
                "Используйте /start для начала работы с ботом.\n"
                "Доступные команды:\n"
                "/start - начать работу\n"
                "/mystack - посмотреть мои данные\n"
                "/update - обновить данные"
            )
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_other: {e}")


@dp.errors()
async def error_handler(event, exception):
    """Глобальный обработчик ошибок"""
    logger.error(f"❌ Глобальная ошибка: {exception}", exc_info=True)
    try:
        if hasattr(event, 'update') and event.update.message:
            await event.update.message.answer(
                "Произошла непредвиденная ошибка. Попробуйте еще раз или используйте /start для перезапуска."
            )
    except Exception as e:
        logger.error(f"❌ Ошибка в error_handler: {e}")


async def set_bot_commands():
    """Установка команд бота"""
    try:
        commands = [
            types.BotCommand(
                command="start", description="🚀 Начать работу с ботом"),
            types.BotCommand(command="mystack",
                             description="📋 Посмотреть мои данные"),
            types.BotCommand(command="update",
                             description="✏️ Обновить данные"),
        ]
        await bot.set_my_commands(commands)
        logger.info("✅ Команды бота успешно установлены")
    except Exception as e:
        logger.error(f"❌ Ошибка установки команд: {e}")


async def on_startup():
    """Действия при запуске бота"""
    try:
        init_db()
        await set_bot_commands()
        logger.info("🚀 Бот успешно запущен!")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при запуске: {e}")
        raise


async def on_shutdown():
    """Действия при остановке бота"""
    logger.info("🛑 Бот остановлен")


async def health_check(request):
    """Endpoint для проверки что бот жив"""
    return web.Response(text="Bot is running!")


async def start_web_server():
    """Запуск веб-сервера для Render"""
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv('PORT', 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"🌐 Веб-сервер запущен на порту {port}")


async def main():
    """Главная функция запуска бота"""
    try:
        await on_startup()

        # Запускаем веб-сервер и бота параллельно
        await asyncio.gather(
            start_web_server(),
            dp.start_polling(bot, on_shutdown=on_shutdown)
        )
    except KeyboardInterrupt:
        logger.info("⚠️ Бот остановлен пользователем")
    except Exception as e:
        logger.critical(f"❌ Критическая ошибка: {e}", exc_info=True)
    finally:
        await bot.session.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"❌ Фатальная ошибка: {e}", exc_info=True)
