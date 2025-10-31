"""Test Tinker API connection and basic training/sampling roundtrip."""

import tinker
from tinker.types import ModelInput, SamplingParams


def main():
    print("Testing Tinker API connection...")
    
    # Test 1: ServiceClient and supported models
    print("\n1. Creating ServiceClient and fetching supported models...")
    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()
    print(f"Supported models: {capabilities.supported_models[:5]}...")  # Show first 5
    
    # Test 2: Create a tiny LoRA training client
    print("\n2. Creating LoRA training client on meta-llama/Llama-3.1-8B-Instruct...")
    training_client = service_client.create_lora_training_client(
        base_model="meta-llama/Llama-3.1-8B-Instruct",  # Using supported model
        rank=32
    )
    print("Training client created successfully")
    
    # Test 3: Save and create sampling client
    print("\n3. Saving weights and creating sampling client...")
    sampling_client = training_client.save_weights_and_get_sampling_client("test_connection")
    print("Sampling client created successfully")
    
    # Test 4: Test sampling
    print("\n4. Testing sampling...")
    tokenizer = training_client.get_tokenizer()
    prompt_text = "Hello, how are you?"
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt = ModelInput.from_ints(prompt_tokens)
    
    result = sampling_client.sample(
        prompt, 
        SamplingParams(max_tokens=10, temperature=0.7), 
        num_samples=1
    )
    sample_result = result.result()
    
    # Decode response
    response_tokens = sample_result.sequences[0].tokens
    response_text = tokenizer.decode(response_tokens)
    
    print(f"Prompt: {prompt_text}")
    print(f"Response: {response_text}")
    
    print("\n✅ All Tinker API tests passed!")


if __name__ == "__main__":
    main()
