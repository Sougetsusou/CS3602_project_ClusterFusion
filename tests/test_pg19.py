from datasets import load_dataset
from tqdm import tqdm

def stream_pg19(split="train", max_samples=5):
    print(f"Loading PG-19 {split} split with streaming...")
    
    # Load dataset with streaming
    dataset = load_dataset("emozilla/pg19", split=split, streaming=True)
    
    # Take a few samples to demonstrate
    sample_count = 0
    for example in dataset:
        if sample_count >= max_samples:
            break
            
        print("\n" + "="*50)
        print(f"Book Title: {example['short_book_title']}")
        print(f"Text Length: {len(example['text'])} characters")
        print("="*50)
        print(example['text'][:500] + "...")  # Print first 500 chars
        
        sample_count += 1

if __name__ == "__main__":
    # Example usage
    stream_pg19(split="train", max_samples=3)
    stream_pg19(split="validation", max_samples=1)
    stream_pg19(split="test", max_samples=1)