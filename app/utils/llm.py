import os
import logging
import sys
from typing import List, Dict, Tuple, Set, Optional, Any
import openai
from openai import OpenAI
import groq
from dotenv import load_dotenv
import time


from app.utils.config import (
    GROUPS,
    GROUP_DESCRIPTIONS,
    GROUP_CATEGORIES,
    GROUP_CATEGORIES_IDS,
    GROUP_IDS,
    GROUP_CATEGORY_DESCRIPTIONS,
    CATEGORIES_IDS
)

# Load environment variables
load_dotenv()

# Get log level from environment (default to INFO)
log_level = getattr(logging, "INFO", logging.INFO)

# Configure logging - console only, no files
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('car_classifier')
logger.info("Logger initialized at level INFO")

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
# llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI if not specified
llm_provider = "openai"

# Initialize clients based on the chosen provider
openai_client = None
groq_client = None

if llm_provider == "openai" and openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized")
elif llm_provider == "groq" and groq_api_key:
    groq_client = groq.Groq(api_key=groq_api_key)
    logger.info("Groq client initialized")
else:
    # Validate API key availability based on the chosen provider
    if llm_provider == "openai" and not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set")
    elif llm_provider == "groq" and not groq_api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        raise ValueError("GROQ_API_KEY environment variable not set")
    else:
        logger.error(f"Invalid LLM_PROVIDER: {llm_provider}. Must be 'openai' or 'groq'")
        raise ValueError(f"Invalid LLM_PROVIDER: {llm_provider}. Must be 'openai' or 'groq'")

# Define models based on provider - use faster models
MODELS = {
    "openai": "gpt-4o",
    "groq": "llama-3.1-8b-instant"  # You can use "mixtral-8x7b-32768" or other Groq models as well
}
logger.info(f"Using {llm_provider} with model {MODELS[llm_provider]}")


def get_completion(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 150,
    system_prompt: Optional[str] = None
) -> str:
    """
    Get a completion from the chosen LLM provider.
    
    Args:
        prompt: The prompt to send to the LLM
        model: Override the default model
        temperature: Temperature parameter for generation
        max_tokens: Maximum number of tokens to generate
        system_prompt: Optional system prompt to set context for the LLM
        
    Returns:
        The generated text
    """
    logger.debug(f"Calling LLM ({llm_provider}) with temperature={temperature}, max_tokens={max_tokens}")
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    start_time = time.time()
    if llm_provider == "openai":
        response = openai_client.chat.completions.create(
            model=model or MODELS["openai"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result = response.choices[0].message.content.strip()
        logger.debug(f"LLM response: {result[:50]}...")
        
    
    elif llm_provider == "groq":
        response = groq_client.chat.completions.create(
            model=model or MODELS["groq"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result = response.choices[0].message.content.strip()
        logger.debug(f"LLM response: {result[:50]}...")
        
    
    else:
        raise ValueError(f"Invalid LLM provider: {llm_provider}")
    
    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Log the information
    logger.info(f"LLM Request Stats: time={time_taken:.2f}s")
    
    return result


def classify_group(text: str) -> str:
    system_prompt = f"""Вы — эксперт по классификации автомобильных проблем. Ваша задача — определить, к какой из указанных сервисных групп относится описание проблемы. Отвечайте только одним из следующих названий групп:
«Новые и разборы», «Детейлинг и тюнинг», «Кузовные работы и детейлинг», «СТО», «Замена масла и жидкостей», «Срочная выездная помощь», «Установка газа на авто».
Не добавляйте пояснений или комментариев.
"""
    
    # If direct matching failed, use LLM classification
    # Create a more descriptive prompt with group descriptions and categories
    prompt = f"""Ниже приведён список сервисных групп, их описаний и примеры категорий (при необходимости, подробности можно найти в полном списке):

**Сервисные группы и описания:**

1. **Новые и разборы**  
   *Описание:* Услуги, связанные с новыми автомобилями и разборками автомобилей.  
   *Примеры категорий:* Расходники/жидкости/фильтры, Аккумуляторы, Салон и детали, Электрика (датчики, моторчики), Трансмиссия АКПП/МКПП/Вариатор, Аксессуары/чехлы/полики, Другое, Мультимедия/акустика/камеры, Стекла, Двигатели и детали, Ходовка, рулевая/тормозная система, Кузов/оптика/свет, Системы охлаждения и кондиционирования.

2. **Детейлинг и тюнинг**  
   *Описание:* Услуги по детейлингу и тюнингу автомобилей, включая косметические и производительные улучшения.  
   *Примеры категорий:* Default Category.

3. **Кузовные работы и детейлинг**  
   *Описание:* Ремонт кузова, покраска, полировка, работы по восстановлению деталей, а также услуги по детейлингу.  
   *Примеры категорий:* Ремонт и покраска кузова/деталей, Сигнализация, Тюнинг кузова и детали из стекловолокна, Покраска и восстановление дисков, Полировка, Удаление вмятин без покраски, Автозвук / Автосвет, Другое, Двигатель, чип тюнинг, Химчистка салона, Шумоизоляция / Перетяжка салона, Бронирование/плёнка/тонировка, 3D полики / Пошив чехлов, Сварочные работы по кузову, Материалы для покраски авто.

4. **СТО**  
   *Описание:* Станции технического обслуживания, предоставляющие общие услуги по ремонту и диагностике автомобилей.  
   *Примеры категорий:* Рулевой механизм, Ремонт печка/кондиционер/радиатор, Ремонт стекол, Ремонт ходовой/подвески/геометрия, Автоэлектрики/компьют диагностика, Чип тюнинг, Ремонт топливной системы, Сварочные/токарные работы, Трансмиссия АКПП/МКПП/Вариатор, Выхлопная система/Ремонт турбин, Ремонт стартера / генератора, Другое, Ремонт/замена двигателя и навесного.

5. **Замена масла и жидкостей**  
   *Описание:* Услуги по замене масел и жидкостей (двигательного масла, трансмиссионной жидкости, тормозной жидкости и проч.).  
   *Примеры категорий:* Замена тормозной жидкости, Замена антифриза, Замена/заправка фреона, Замена масла в коробке АКПП/МКПП, Замена масла в двигателе, Другое.

6. **Срочная выездная помощь**  
   *Описание:* Услуги экстренной выездной помощи, включая ремонт на месте и эвакуацию.  
   *Примеры категорий:* Автоэлектрик на выезд, Ремонт/замена стекол, Трезвый водитель, Эвакуатор, Выездной шиномонтаж, Прикурить, Отогрев, Вскрытие замков, Изготовление ключей.

7. **Установка газа на авто**  
   *Описание:* Услуги по установке и ремонту газобаллонного оборудования для автомобилей.  
   *Примеры категорий:* Установка газобаллонного оборудования, Ремонт газобаллонного оборудования.

---

**Описание проблемы:**  
"{text}"

На основе приведённой информации выберите только ту сервисную группу, которая наиболее точно описывает техническую проблему, игнорируя лишние детали (например, просьбы «не звонить» или указания цены). Ответьте **только** названием группы (одним из: "Новые и разборы", "Детейлинг и тюнинг", "Кузовные работы и детейлинг", "СТО", "Замена масла и жидкостей", "Срочная выездная помощь", "Установка газа на авто").
"""
    
    predicted_group = get_completion(
        prompt, 
        temperature=0.1,  # Lower temperature for more deterministic results
        max_tokens=50,    # Reduced max tokens since we only need the group name
        system_prompt=system_prompt
    )
    
    # Clean up response and handle potential formatting issues
    predicted_group = predicted_group.strip().strip('"\'').split('\n')[0]
    
    # Validate that the predicted group is in our defined groups
    if predicted_group not in GROUPS:
        # Try to find a partial match
        for group in GROUPS:
            if group.lower() in predicted_group.lower():
                return group
        
        # Default to most common group if prediction is invalid
        return "СТО"
    
    return predicted_group


def identify_categories(text: str, group: str) -> List[str]:
    group_categories = list(GROUP_CATEGORIES.get(group, {}).keys())
    
    if not group_categories:
        return ["Default Category"]

    # Define a system prompt that clearly sets expectations
    system_prompt = f"""Вы — эксперт по классификации автомобильных проблем. Ваша задача — сначала понять, что написал клиент и определить, какие из приведённых категорий внутри группы «{group}» соответствуют описанию проблемы. учтите, что текст может быть неформальным, содержать опечатки, избыточные детали. Отвечайте только идентификаторами категорий (через запятую, если их несколько) из списка. Не добавляйте пояснений или комментариев."""

    prompt = f"""Ниже приведён список категорий для группы «{group}» с их идентификаторами и краткими описаниями:"""

    has_descriptions = group in GROUP_CATEGORY_DESCRIPTIONS
    
    # Add each category with its description to the prompt
    for category in group_categories:
        if category == "Default Category":
            continue
            
        # Add description if available
        if has_descriptions and category in GROUP_CATEGORY_DESCRIPTIONS[group]:
            description = GROUP_CATEGORY_DESCRIPTIONS[group][category]
            prompt += f"{GROUP_CATEGORIES_IDS[group][category]} - **{category}**: {description}\n"
        else:
            prompt += f"{GROUP_CATEGORIES_IDS[group][category]} - **{category}**\n"

    prompt += f"""
Описание проблемы:
"{text}"

На основе приведённой информации выберите только те категории, которые наиболее точно описывают техническую проблему. Если проблема относится к нескольким категориям, укажите их через запятую (не более двух-трёх наиболее релевантных вариантов).

Ответьте только идентификаторами категорий, через запятую.
"""

    predicted_categories_text = get_completion(
        prompt, 
        temperature=0.2, 
        max_tokens=100,
        system_prompt=system_prompt
    )

    logger.info(f"LLM predicted categories: {predicted_categories_text}")
    
    # Parse the response
    predicted_categories = []

    for category_id in predicted_categories_text.split(","):
        category = CATEGORIES_IDS.get(category_id.strip().strip('"\'- '), "")
        if category:
            logger.info(f"category: {category}")
            predicted_categories.append(category)
    
    # Validate that the predicted categories are in our defined categories
    valid_categories = [
        category for category in predicted_categories 
        if category in group_categories
    ]
    logger.info(f"valid_categories: {valid_categories}")

    # Return "Другое" (Other) if no valid categories were found
    if not valid_categories and "Другое" in group_categories:
        return {"-1": "Другое"}, ["-1"]
    elif not valid_categories:
        return {"-1": "Другое"}, ["-1"]
    
    # Remove duplicates while preserving order
    final_categories = {}
    final_categories_ids = []
    for category in valid_categories:
        if category in GROUP_CATEGORIES_IDS[group]:
            final_categories[GROUP_CATEGORIES_IDS[group][category]] = category
            final_categories_ids.append(GROUP_CATEGORIES_IDS[group][category])
    
    return final_categories, final_categories_ids


def classify_car_issue(text: str) -> Tuple[str, List[str]]:
    """
    Classify a car issue text into a group and categories within that group.
    
    Args:
        text: The car issue description text
        
    Returns:
        Tuple of (group name, list of category names)
    """
    # First classify the group
    group = classify_group(text)
    logger.info(f"group: {group}")
    # Then identify categories within that group
    categories, categories_ids = identify_categories(text, group)
    
    return group, GROUP_IDS.get(group, -1), categories_ids, categories
