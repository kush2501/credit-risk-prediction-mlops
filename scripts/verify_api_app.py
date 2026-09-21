from src.api.main import app


print("FastAPI application imported successfully!")

print("\nApplication title:")
print(app.title)

print("\nApplication version:")
print(app.version)

print("\nAvailable routes:")

for route in app.routes:
    print(
        route.path,
        route.methods
    )