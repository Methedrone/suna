<div align="center">

# Suna - Open Source Generalist AI Agent

(that acts on your behalf)

![Suna Screenshot](frontend/public/banner.png)

Suna is a fully open source AI assistant that helps you accomplish real-world tasks with ease. Through natural conversation, Suna becomes your digital companion for research, data analysis, and everyday challenges—combining powerful capabilities with an intuitive interface that understands what you need and delivers results.

Suna's powerful toolkit includes seamless browser automation to navigate the web and extract data, file management for document creation and editing, web crawling and extended search capabilities, command-line execution for system tasks, website deployment, and integration with various APIs and services. These capabilities work together harmoniously, allowing Suna to solve your complex problems and automate workflows through simple conversations!

[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](./license)
[![Discord Follow](https://dcbadge.limes.pink/api/server/Py6pCBUUPw?style=flat)](https://discord.gg/Py6pCBUUPw)
[![Twitter Follow](https://img.shields.io/twitter/follow/kortixai)](https://x.com/kortixai)
[![GitHub Repo stars](https://img.shields.io/github/stars/kortix-ai/suna)](https://github.com/kortix-ai/suna)
[![Issues](https://img.shields.io/github/issues/kortix-ai/suna
)](https://github.com/kortix-ai/suna/labels/bug)
</div>


## Table of Contents

- [Suna Architecture](#project-architecture)
  - [Backend API](#backend-api)
  - [Frontend](#frontend)
  - [Agent Docker](#agent-docker)
  - [Supabase Database](#supabase-database)
- [Run Locally / Self-Hosting](#run-locally--self-hosting)
  - [Requirements](#requirements)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Architecture

![Architecture Diagram](docs/images/diagram.png)

Suna consists of four main components:

### Backend API
Python/FastAPI service that handles REST endpoints, thread management, and LLM integration with Anthropic, and others via LiteLLM.

### Frontend
Next.js/React application providing a responsive UI with chat interface, dashboard, etc.

### Agent Docker
Isolated execution environment for every agent - with browser automation, code interpreter, file system access, tool integration, and security features.

### Supabase Database
Handles data persistence with authentication, user management, conversation history, file storage, agent state, analytics, and real-time subscriptions.

## Use Cases

1. **Competitor Analysis** ([Watch](https://www.suna.so/share/5ee791ac-e19c-4986-a61c-6d0659d0e5bc)) - *"Analyze the market for my next company in the healthcare industry, located in the UK. Give me the major players, their market size, strengths, and weaknesses, and add their website URLs. Once done, generate a PDF report."*

2. **VC List** ([Watch](https://www.suna.so/share/804d20a3-cf1c-4adb-83bb-0e77cc6adeac)) - *"Give me the list of the most important VC Funds in the United States based on Assets Under Management. Give me website URLs, and if possible an email to reach them out."*

3. **Looking for Candidates** ([Watch](https://www.suna.so/share/3ae581b0-2db8-4c63-b324-3b8d29762e74)) - *"Go on LinkedIn, and find me 10 profiles available - they are not working right now - for a junior software engineer position, who are located in Munich, Germany. They should have at least one bachelor's degree in Computer Science or anything related to it, and 1-year of experience in any field/role."*

4. **Planning Company Trip** ([Watch](https://www.suna.so/share/725e64a0-f1e2-4bb6-8a1f-703c2833fd72)) - *"Generate me a route plan for my company. We should go to California. We'll be in 8 people. Compose the trip from the departure (Paris, France) to the activities we can do considering that the trip will be 7 days long - departure on the 21st of Apr 2025. Check the weather forecast and temperature for the upcoming days, and based on that, you can plan our activities (outdoor vs indoor)."*

5. **Working on Excel** ([Watch](https://www.suna.so/share/128f23a4-51cd-42a6-97a0-0b458b32010e)) - *"My company asked me to set up an Excel spreadsheet with all the information about Italian lottery games (Lotto, 10eLotto, and Million Day). Based on that, generate and send me a spreadsheet with all the basic information (public ones)."*

6. **Automate Event Speaker Prospecting** ([Watch](https://www.suna.so/share/7a7592ea-ed44-4c69-bcb5-5f9bb88c188c)) - *"Find 20 AI ethics speakers from Europe who've spoken at conferences in the past year. Scrapes conference sites, cross-references LinkedIn and YouTube, and outputs contact info + talk summaries."*

7. **Summarize and Cross-Reference Scientific Papers** ([Watch](https://www.suna.so/share/c2081b3c-786e-4e7c-9bf4-46e9b23bb662)) - *"Research and compare scientific papers talking about Alcohol effects on our bodies during the last 5 years. Generate a report about the most important scientific papers talking about the topic I wrote before."*

8. **Research + First Contact Draft** ([Watch](https://www.suna.so/share/6b6296a6-8683-49e5-9ad0-a32952d12c44)) - *"Research my potential customers (B2B) on LinkedIn. They should be in the clean tech industry. Find their websites and their email addresses. After that, based on the company profile, generate a personalized first contact email where I present my company which is offering consulting services to cleantech companies to maximize their profits and reduce their costs."*

9. **SEO Analysis** ([Watch](https://www.suna.so/share/43491cb0-cd6c-45f0-880c-66ddc8c4b842)) - *"Based on my website suna.so, generate an SEO report analysis, find top-ranking pages by keyword clusters, and identify topics I'm missing."*

10. **Generate a Personal Trip** ([Watch](https://www.suna.so/share/37b31907-8349-4f63-b0e5-27ca597ed02a)) - *"Generate a personal trip to London, with departure from Bangkok on the 1st of May. The trip will last 10 days. Find an accommodation in the center of London, with a rating on Google reviews of at least 4.5. Find me interesting outdoor activities to do during the journey. Generate a detailed itinerary plan."*

11. **Recently Funded Startups** ([Watch](https://www.suna.so/share/8b2a897e-985a-4d5e-867b-15239274f764)) - *"Go on Crunchbase, Dealroom, and TechCrunch, filter by Series A funding rounds in the SaaS Finance Space, and build a report with company data, founders, and contact info for outbound sales."*

12. **Scrape Forum Discussions** ([Watch](https://www.suna.so/share/7d7a5d93-a20d-48b0-82cc-e9a876e9fd04)) - *"I need to find the best beauty centers in Rome, but I want to find them by using open forums that speak about this topic. Go on Google, and scrape the forums by looking for beauty center discussions located in Rome. Then generate a list of 5 beauty centers with the best comments about them."*

## Run Locally / Self-Hosting

Suna can be self-hosted on your own infrastructure. Follow these steps to set up your own instance.

### Requirements

You'll need the following components:
- A Supabase project for database and authentication
- Redis database for caching and session management
- Daytona sandbox for secure agent execution
- Python 3.11 for the API backend
- API keys for LLM providers (Anthropic, OpenRouter)
- Tavily API key for enhanced search capabilities
- Firecrawl API key for web scraping capabilities

### Prerequisites

1. **Supabase**:
   - Create a new [Supabase project](https://supabase.com/dashboard/projects)
   - Save your project's API URL, anon key, and service role key for later use
   - Install the [Supabase CLI](https://supabase.com/docs/guides/cli/getting-started)

2. **Redis**: 
   - Go to the `/backend` folder
   - Run `docker compose up redis`

3. **Daytona**:
   - Create an account on [Daytona](https://app.daytona.io/)
   - Generate an API key from your account settings
   - Go to [Images](https://app.daytona.io/dashboard/images)
   - Click "Add Image"
   - Enter `kortix/suna:0.1.2` as the image name
   - Set `/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf` as the Entrypoint

4. **LLM API Keys**:
   - Obtain an API key [Anthropic](https://www.anthropic.com/)
   - While other providers should work via [LiteLLM](https://github.com/BerriAI/litellm), Anthropic is recommended – the prompt needs to be adjusted for other providers to output correct XML for tool calls.

5. **Search API Key** (Optional):
   - For enhanced search capabilities, obtain an [Tavily API key](https://tavily.com/)
   - For web scraping capabilities, obtain a [Firecrawl API key](https://firecrawl.dev/)
  

6. **RapidAPI API Key** (Optional):
   - To enable API services like LinkedIn, and others, you'll need a RapidAPI key
   - Each service requires individual activation in your RapidAPI account:
     1. Locate the service's `base_url` in its corresponding file (e.g., `"https://linkedin-data-scraper.p.rapidapi.com"` in [`backend/agent/tools/data_providers/LinkedinProvider.py`](backend/agent/tools/data_providers/LinkedinProvider.py))
     2. Visit that specific API on the RapidAPI marketplace
     3. Subscribe to the service (many offer free tiers with limited requests)
     4. Once subscribed, the service will be available to your agent through the API Services tool

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/kortix-ai/suna.git
cd suna
```

2. **Configure backend environment**:
```bash
cd backend
cp .env.example .env  # Create from example if available, or use the following template
```

Edit the `.env` file and fill in your credentials:
```bash
NEXT_PUBLIC_URL="http://localhost:3000"

# Supabase credentials from step 1
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Redis credentials from step 2
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
REDIS_SSL=True  # Set to False for local Redis without SSL

# Daytona credentials from step 3
DAYTONA_API_KEY=your_daytona_api_key
DAYTONA_SERVER_URL="https://app.daytona.io/api"
DAYTONA_TARGET="us"

# Anthropic
ANTHROPIC_API_KEY=

# OpenAI API:
OPENAI_API_KEY=your_openai_api_key

# Google API:
GOOGLE_API_KEY=your_google_api_key

# Optional but recommended
TAVILY_API_KEY=your_tavily_api_key  # For enhanced search capabilities
FIRECRAWL_API_KEY=your_firecrawl_api_key  # For web scraping capabilities
RAPID_API_KEY=
```

3. **Set up Supabase database**:
```bash
# Login to Supabase CLI
supabase login

# Link to your project (find your project reference in the Supabase dashboard)
supabase link --project-ref your_project_reference_id

# Push database migrations
supabase db push
```

Then, go to the Supabase web platform again -> choose your project -> Project Settings -> Data API -> And in the "Exposed Schema" add "basejump" if not already there

4. **Configure frontend environment**:
```bash
cd ../frontend
cp .env.example .env.local  # Create from example if available, or use the following template
```

   Edit the `.env.local` file:
```
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_BACKEND_URL="http://localhost:8000/api"  # Use this for local development
NEXT_PUBLIC_URL="http://localhost:3000"
```

   Note: If you're using Docker Compose, use the container name instead of localhost:
```
NEXT_PUBLIC_BACKEND_URL="http://backend:8000/api"  # Use this when running with Docker Compose
```

5. **Install dependencies**:
```bash
# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd ../backend
poetry install
```

6. **Start the application**:

   In one terminal, start the frontend:
```bash
cd frontend
npm run dev
```

   In another terminal, start the backend:
```bash
cd backend
poetry run python3.11 api.py
```

5-6. **Docker Compose Alternative**:

Before running with Docker Compose, make sure your environment files are properly configured:
- In `backend/.env`, set all the required environment variables as described above
  - For Redis configuration, use `REDIS_HOST=redis` instead of localhost
  - The Docker Compose setup will automatically set these Redis environment variables:
    ```
    REDIS_HOST=redis
    REDIS_PORT=6379
    REDIS_PASSWORD=
    REDIS_SSL=False
    ```
- In `frontend/.env.local`, make sure to set `NEXT_PUBLIC_BACKEND_URL="http://backend:8000/api"` to use the container name

Then run:
```bash
export GITHUB_REPOSITORY="your-github-username/repo-name"
docker compose -f docker-compose.ghcr.yaml up
```

If you're building the images locally instead of using pre-built ones:
```bash
docker compose up
```

The Docker Compose setup includes a Redis service that will be used by the backend automatically.


7. **Access Suna**:
   - Open your browser and navigate to `http://localhost:3000`
   - Sign up for an account using the Supabase authentication
   - Start using your self-hosted Suna instance!

## Acknowledgements

### Main Contributors
- [Adam Cohen Hillel](https://x.com/adamcohenhillel)
- [Dat-lequoc](https://x.com/datlqqq)
- [Marko Kraemer](https://twitter.com/markokraemer)

### Technologies
- [Daytona](https://daytona.io/) - Secure agent execution environment
- [Supabase](https://supabase.com/) -
- [Playwright](https://playwright.dev/) - Browser automation
- [OpenAI](https://openai.com/) - LLM provider
- [Anthropic](https://www.anthropic.com/) - LLM provider
- [Tavily](https://tavily.com/) - Search capabilities
- [Firecrawl](https://firecrawl.dev/) - Web scraping capabilities
- [RapidAPI](https://rapidapi.com/) - API services


## License

Kortix Suna is licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) for the full license text.
