# Car Issue Classification API

This API classifies car issues based on text descriptions into specific groups and categories using LLM. It is designed to be integrated into car garage platforms.

## Features

- Classify car issue descriptions into specific groups
- Further categorize issues within the group
- Support for both OpenAI and Groq LLM providers
- Enhanced LLM classification
- Simple, focused REST API with a single classification endpoint
- Ready to deploy on fly.dev

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API Key or Groq API Key

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd car-issue-classifier
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API key and preferred LLM provider:
```
# Choose one of the following options:

# Option 1: Using OpenAI (default)
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai

# Option 2: Using Groq
# GROQ_API_KEY=your_groq_api_key_here
# LLM_PROVIDER=groq
```

5. Run the application:
```bash
python run.py
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /api/v1/classify

Classifies a car issue text into a group and categories.

**Request Body**:
```json
{
    "text": "Нужно заменить масло в двигателе и проверить тормозную жидкость"
}
```

Parameters:
- `text` (required): The car issue description text

**Response**:
```json
{
    "group": "Замена масла и жидкостей",
    "group_id": 25,
    "categories": {},
    "categories_ids": [],
    "method_used": "hashtag_match"
}
```

### GET /api/v1/health

Health check endpoint to verify the API is running.

**Response**:
```json
{
    "status": "healthy"
}
```

## Deployment to fly.dev

1. Install the Fly CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Login to Fly:
```bash
fly auth login
```

3. Launch the app (first-time deployment):
```bash
fly launch
```

4. Set your API keys and provider as secrets:
```bash
# For OpenAI
fly secrets set OPENAI_API_KEY=your_openai_api_key_here LLM_PROVIDER=openai

# For Groq
fly secrets set GROQ_API_KEY=your_groq_api_key_here LLM_PROVIDER=groq
```

5. Deploy updates to the app:
```bash
fly deploy
```

## Usage Example

### Python Example

```python
import requests

url = "https://your-app-name.fly.dev/api/v1/classify"
payload = {
    "text": "Нужно заменить масло в двигателе и проверить тормозную жидкость"
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### Curl Example

Test the classification endpoint:
```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Сломалась рулевая рейка, течет жидкость"
  }'
```

## License

MIT 