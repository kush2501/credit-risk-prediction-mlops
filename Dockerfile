# Use Python 3.11 with a small Linux operating system
FROM python:3.11-slim


# Set /app as the working directory inside the Docker container
WORKDIR /app


# Copy the requirements folder from the local project into the Docker container
COPY requirements/ requirements/


# Install the basic Python dependencies
# --no-cache-dir keeps the Docker image smaller by not saving pip's download cache
RUN pip install --no-cache-dir -r requirements/base.txt \
    && pip install --no-cache-dir -r requirements/ml.txt \
    && pip install --no-cache-dir -r requirements/tracking.txt \
    && pip install --no-cache-dir -r requirements/api.txt


# Copy the application source code into the Docker container
COPY src/ src/


# Copy the configuration files into the Docker container
COPY config/ config/

# Copy the preprocessing artifact required by the API
COPY artifacts/data_transformation/preprocessor.pkl artifacts/data_transformation/preprocessor.pkl

# Tell Docker that the FastAPI application uses port 8000
EXPOSE 8000


# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]