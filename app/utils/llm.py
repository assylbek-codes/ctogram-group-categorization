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
llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI if not specified

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
    "openai": "gpt-3.5-turbo",
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
    """
    Use the chosen LLM to classify the car issue text into one of the predefined groups.
    
    Args:
        text: The car issue description text
        
    Returns:
        The classified group name
    """
    # First try to match directly using hashtags
    for group, categories_dict in GROUP_CATEGORIES.items():
        for category, hashtags in categories_dict.items():
            if not hashtags:
                continue
                
            # Clean hashtags for matching
            clean_hashtags = [tag.lower().replace('#', '') for tag in hashtags]
            
            # Check if any hashtags appear in the text
            for tag in clean_hashtags:
                if tag in text.lower():
                    # Found a direct match to a hashtag
                    return group
    
    # Define a system prompt that clearly sets expectations
    system_prompt = f"""You are an expert automotive service classifier.
Your task is to classify car repair issues into the correct service group.
You have deep knowledge of car components, repair procedures, and automotive terminology in Russian.
You will be presented with a car issue description and must determine which service group it belongs to.
You will respond ONLY with the exact group name from the provided list - nothing else.
"""
    
    # If direct matching failed, use LLM classification
    # Create a more descriptive prompt with group descriptions and categories
    prompt = f"""I need to classify this car issue description into one of the following service groups:

"""
    
    # Add each group with its description and categories to the prompt
    for group, description in GROUP_DESCRIPTIONS.items():
        prompt += f"- {group}: {description}\n"
        
        # Add category examples for each group to help with classification
        if group in GROUP_CATEGORIES:
            categories = list(GROUP_CATEGORIES[group].keys())
            # Only list non-default categories
            example_categories = [c for c in categories if c != "Default Category"]
            
            if len(example_categories) > 0:
                # Only list up to 5 categories to keep the prompt manageable
                display_categories = example_categories[:4] if len(example_categories) > 4 else example_categories
                prompt += f"  Example categories: {', '.join(display_categories)}\n"
                
                # # Add some example hashtags from each category
                # all_hashtags = []
                # for category in example_categories:
                #     category_hashtags = GROUP_CATEGORIES[group].get(category, [])
                #     if category_hashtags:
                #         # Take just a few hashtags from each category
                #         all_hashtags.extend(category_hashtags[:3])
                
                # if all_hashtags:
                #     # Only show a reasonable number of hashtags in total
                #     display_hashtags = all_hashtags[:10] if len(all_hashtags) > 10 else all_hashtags
                #     prompt += f"  Example keywords: {', '.join(display_hashtags)}\n"
    
    prompt += f"""
Car issue description: "{text}"

Which ONE of the above groups does this issue belong to? Respond with ONLY the exact group name - nothing else.
"""
    
    predicted_group = get_completion(
        prompt, 
        temperature=0.2,  # Lower temperature for more deterministic results
        max_tokens=20,    # Reduced max tokens since we only need the group name
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
    """
    Use the chosen LLM to identify relevant categories based on the car issue text and group.
    
    Args:
        text: The car issue description text
        group: The classified group name
        
    Returns:
        List of category names within the group
    """
    # Get categories for the classified group
    group_categories = list(GROUP_CATEGORIES.get(group, {}).keys())
    
    if not group_categories:
        return ["Default Category"]

    
    # Define a system prompt that clearly sets expectations
    system_prompt = f"""You are an expert automotive service classifier for a car garage platform.
Your task is to categorize car repair issues into the most appropriate categories within a specific service group.
You have deep knowledge of car components, repair procedures, and automotive terminology in Russian.
You will be presented with a car issue description and must identify which categories within a given group it belongs to.
You will respond ONLY with the exact category names, comma-separated if multiple categories apply.
Do not include ANY explanations, notes, or other text in your response - only the category names.
"""
    
    # Otherwise use LLM classification
    prompt = f"""I need to categorize this car issue description into categories within the "{group}" group.

Group: {group} - {GROUP_DESCRIPTIONS.get(group, "")}

Available categories within this group:
"""
    
    # Add detailed information for each category
    # Add categories with descriptions if available
    has_descriptions = group in GROUP_CATEGORY_DESCRIPTIONS
    
    # Add each category with its description to the prompt
    for category in group_categories:
        if category == "Default Category":
            continue
            
        # Add description if available
        if has_descriptions and category in GROUP_CATEGORY_DESCRIPTIONS[group]:
            description = GROUP_CATEGORY_DESCRIPTIONS[group][category]
            prompt += f"- {category}: {description}\n"
        else:
            prompt += f"- {category}\n"
            
        # # Add hashtags
        # hashtags = GROUP_CATEGORIES[group].get(category, [])
        # if hashtags and len(hashtags) > 0:
        #     # Only show a few hashtags as examples to keep the prompt manageable
        #     example_hashtags = hashtags[:10] if len(hashtags) > 10 else hashtags
        #     prompt += f"  Related keywords: {', '.join(example_hashtags)}\n"
    
    prompt += f"""
Car issue description: "{text}"

Which of the above categories does this issue belong to? Respond with ONLY the exact category names, comma-separated if multiple apply.
"""
    
    predicted_categories_text = get_completion(
        prompt, 
        temperature=0.3, 
        max_tokens=100,
        system_prompt=system_prompt
    )
    
    # Parse the response and handle potential formatting issues
    predicted_categories = [
        category.strip().strip('"\'- ') 
        for category in predicted_categories_text.split(",")
    ]
    
    # Validate that the predicted categories are in our defined categories
    valid_categories = [
        category for category in predicted_categories 
        if category in group_categories
    ]
    
    # If we didn't find any valid categories, try to find partial matches
    if not valid_categories:
        for predicted in predicted_categories:
            for valid in group_categories:
                if predicted in valid or valid in predicted:
                    valid_categories.append(valid)
                    break
    
    # Return "Другое" (Other) if no valid categories were found
    if not valid_categories and "Другое" in group_categories:
        return ["Другое"]
    elif not valid_categories:
        return ["Default Category"]
    
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
    
    # Then identify categories within that group
    categories, categories_ids = identify_categories(text, group)
    
    return group, GROUP_IDS.get(group, -1), categories_ids, categories


def classify_car_issue_categories_first(text: str) -> Tuple[str, List[str]]:
    """
    Alternative approach that first identifies potential categories across all groups
    and then determines the appropriate group based on the categories.
    This can be more accurate for cases where category recognition is easier than group recognition.
    
    Args:
        text: The car issue description text
        
    Returns:
        Tuple of (group name, list of category names)
    """
    logger.info(f"Starting categories-first LLM classification for: '{text[:50]}...'")

    # Create a mapping of all categories to their groups
    all_categories = {}
    for group, categories_dict in GROUP_CATEGORIES.items():
        for category in categories_dict.keys():
            all_categories[category] = group
    
    # Define a system prompt that clearly sets expectations
    system_prompt = f"""You are an expert automotive service classifier for a car garage platform.
You have deep knowledge of car repair services, components, and automotive terminology in Russian.
Your task is to analyze car issue descriptions and classify them into the most appropriate service categories.
You understand the distinction between different types of services (repairs, replacements, maintenance).
You will respond ONLY with the exact category number, comma-separated if multiple categories apply.
Do not include ANY explanations, notes, or other text in your response - only the category numbers.
"""
    
    # Create a prompt to identify potential categories across all groups
    prompt = f"""I need to identify the most appropriate service categories for the following car issue description.

Important classification rules:
1. Repairs to components (like radiators, engines, transmissions) belong to repair services, not fluid services
2. "Замена" (replacement) of parts is different from "Замена" of fluids - pay attention to what is being replaced
3. If specific components are mentioned, prioritize the component-specific category over general categories
4. Technical terms and specific parts mentioned should guide your classification

Available categories by group:
"""
    # Add categories from all groups with their hashtags
    for group, categories_dict in GROUP_CATEGORIES.items():
        # Add group name and description
        group_desc = GROUP_DESCRIPTIONS.get(group, "")
        prompt += f"\n## Categories from {group} ({group_desc}):\n"
        
        # Add categories for this group
        for category, hashtags in categories_dict.items():
            if category == "Default Category":
                continue
                
            # Add category with description if available
            category_desc = ""
            if group in GROUP_CATEGORY_DESCRIPTIONS and category in GROUP_CATEGORY_DESCRIPTIONS[group]:
                category_desc = GROUP_CATEGORY_DESCRIPTIONS[group][category]
                prompt += f"{GROUP_CATEGORIES_IDS[group][category]} - {category}: {category_desc}\n"
            else:
                prompt += f"{GROUP_CATEGORIES_IDS[group][category]} - {category}\n"
            
            # # Add hashtags if available
            # if hashtags and len(hashtags) > 0:
            #     # Show more hashtags for better coverage
            #     example_hashtags = hashtags[:15] if len(hashtags) > 15 else hashtags
            #     prompt += f"  Keywords: {', '.join(example_hashtags)}\n"
    
    prompt += f"""
Car issue description: "{text}"

Which categories best match this car issue? Respond with ONLY the EXACT FULL category numbers from the given list above, comma-separated if multiple apply.
"""
    
    predicted_categories_text = get_completion(
        prompt, 
        temperature=0.2, 
        max_tokens=150,
        system_prompt=system_prompt
    )
    logger.info(f"LLM predicted categories: {predicted_categories_text}")
    
    # Parse the response
    predicted_categories = []

    for category_id in predicted_categories_text.split(","):
        category = CATEGORIES_IDS.get(category_id.strip().strip('"\'- '), "")
        if category:
            predicted_categories.append(category)
    
    # Match categories to valid ones and count groups
    valid_categories = []
    group_counts = {}
    
    # First try exact matches
    for predicted in predicted_categories:
        if predicted in all_categories:
            valid_categories.append(predicted)
            group = all_categories[predicted]
            group_counts[group] = group_counts.get(group, 0) + 1
            logger.debug(f"Exact match for category: '{predicted}' -> group: '{group}'")
        else:
            logger.debug(f"No match for category: '{predicted}'")
    
    # If no exact matches, try partial matches
    if not valid_categories:
        logger.info("No exact category matches found, trying partial matches")
        for predicted in predicted_categories:
            for valid, group in all_categories.items():
                if predicted.lower() in valid.lower() or valid.lower() in predicted.lower():
                    valid_categories.append(valid)
                    group_counts[group] = group_counts.get(group, 0) + 1
                    logger.debug(f"Partial match: '{predicted}' ~ '{valid}' -> group: '{group}'")
                    break
    
    # Determine the most likely group based on category counts
    if group_counts:
        logger.info(f"Group counts from categories: {group_counts}")
        likely_group = max(group_counts.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected group: '{likely_group}' with {group_counts[likely_group]} category matches")
    else:
        # Fall back to the standard group classification
        logger.info("No categories matched, falling back to standard classification")
        return classify_car_issue(text)
    
    # Filter categories to only include those from the identified group
    final_categories = {}
    final_categories_ids = []
    for category in valid_categories:
        if category in all_categories and all_categories.get(category) == likely_group:
            final_categories[GROUP_CATEGORIES_IDS[likely_group][category]] = category
            final_categories_ids.append(GROUP_CATEGORIES_IDS[likely_group][category])

    logger.info(f"Initial categories for '{likely_group}': {final_categories}")
    
    # If we didn't find any categories for the identified group, get them using the standard approach
    if not final_categories:
        logger.info(f"No categories found for group '{likely_group}', using identify_categories")
        final_categories, final_categories_ids = identify_categories(text, likely_group)

    logger.info(f"Final categories-first result: Group='{likely_group}', Categories={final_categories}")
    return likely_group, GROUP_IDS.get(likely_group, -1), final_categories_ids, final_categories


def classify_car_issue_with_hashtags(text: str) -> Tuple[str, List[str]]:
    logger.info(f"Starting classification for: '{text[:50]}...'")
    # Input validation
    if not text or len(text) < 3:
        return "Другое", GROUP_IDS.get("Другое", -1), [], {}
    
    return classify_car_issue_categories_first(text) 