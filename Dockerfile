# Step 1: Use an official Python runtime as a parent image
FROM python:3.12-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements.txt file into the container
COPY requirements.txt .

# Step 4: Install any needed dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the current directory contents into the container
COPY . .

# Step 6: Install Uvicorn (if not already in requirements.txt)
RUN pip install uvicorn

# Step 7: Expose the port your FastAPI app will run on (default is 8000)
EXPOSE 8000

# Step 8: Run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
