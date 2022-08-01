FROM python:3.9

WORKDIR /project

# Copy over contents from local directory to the path in Docker container
COPY requirements.txt /project/requirements.txt
# COPY . /project/

# Install python requirements from requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . /project/

WORKDIR /project/app

# Start uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]