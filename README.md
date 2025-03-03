# PDF Title Generator

Appealmate-AI is a Flask-based application designed to process various tasks using worker processes. It utilizes Gunicorn for production deployment and integrates with MongoDB for data storage. The system supports administrative and user task processing via separate workers.

## Features
- Flask-based web application
- Asynchronous worker processes for user and admin tasks
- MongoDB integration
- Configurable logging and monitoring
- Gunicorn-based deployment for production

## Prerequisites

- Python 3.x
- AWS SDK for Python (Boto3)
- PyPDF2
- ReportLab
- OpenAI's GPT model (via `autogen` or similar)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-title-generator.git
   cd pdf-title-generator

## Usage

### Running in Development Mode
Start the Flask application with:
```sh
python run.py```
This runs the server on port 8000 with debugging enabled.

### Running in Production Mode

Use Gunicorn to serve the application:

```
gunicorn -c gunicorn.conf.py prod_run:app```

### Running Workers

Start the user worker:
```
python run_worker.py ```

Start the admin worker:
``` 
python run_admin_worker.py ```
