from src.utils.common import save_object


data = {
    "name": "Rahul",
    "age": 25,
    "city": "Delhi",
}

save_object(
    "artifacts/test_object.pkl",
    data,
)

print("Object saved successfully.")