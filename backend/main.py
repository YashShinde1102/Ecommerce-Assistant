#fast api entry points 
from pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    query = input("Enter query: ")
    image_path = input("Enter image path (optional): ")

    image_path = image_path if image_path.strip() else None

    answer = run_pipeline(query, image_path)
    print("\nAnswer:\n", answer)
