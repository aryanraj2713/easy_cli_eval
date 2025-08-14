

from typing import Any, Dict, List, Optional

from ...core.exceptions import DatasetError

def load_dataset(dataset_name: str, num_samples: int = 10) -> List[Dict[str, Any]]:
    
    
    try:
        if dataset_name == "squad_v2":
            return _load_squad_samples(num_samples)
        elif dataset_name == "cnn_dailymail":
            return _load_cnn_dailymail_samples(num_samples)
        elif dataset_name == "wmt16":
            return _load_wmt16_samples(num_samples)
        elif dataset_name == "glue":
            return _load_glue_samples(num_samples)
        else:
            return _load_generic_samples(num_samples)
            
    except Exception as e:
        raise DatasetError(f"Failed to load dataset '{dataset_name}': {str(e)}")

def _load_squad_samples(num_samples: int) -> List[Dict[str, Any]]:
    
    samples = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris, which is known for its art, culture, and history.",
            "answer": "Paris",
        },
        {
            "question": "Who wrote the novel 'Pride and Prejudice'?",
            "context": "Pride and Prejudice is a romantic novel by Jane Austen, published in 1813. It follows the character development of Elizabeth Bennet.",
            "answer": "Jane Austen",
        },
        {
            "question": "What year did World War II end?",
            "context": "World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries, including all of the great powers.",
            "answer": "1945",
        },
        {
            "question": "What is the largest planet in our solar system?",
            "context": "Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun.",
            "answer": "Jupiter",
        },
        {
            "question": "Who painted the Mona Lisa?",
            "context": "The Mona Lisa is a half-length portrait painting by the Italian Renaissance artist Leonardo da Vinci. It is considered an archetypal masterpiece of the Italian Renaissance.",
            "answer": "Leonardo da Vinci",
        },
    ]
    
    while len(samples) < num_samples:
        samples.extend(samples[:num_samples - len(samples)])
    
    return samples[:num_samples]

def _load_cnn_dailymail_samples(num_samples: int) -> List[Dict[str, Any]]:
    
    samples = [
        {
            "text": "The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one of the most famous paintings in the world. It is housed in the Louvre Museum in Paris, France. The painting is known for its subtle smile and mysterious gaze that seems to follow viewers around the room. It has been the subject of numerous studies, theories, and even theft attempts throughout history.",
            "summary": "The Mona Lisa is a famous painting by Leonardo da Vinci housed in the Louvre Museum in Paris, known for its mysterious smile and gaze.",
        },
        {
            "text": "Artificial intelligence continues to advance at a rapid pace, with new breakthroughs being announced regularly. Machine learning models are becoming increasingly sophisticated, able to perform tasks that once required human intelligence. However, concerns about AI safety, ethics, and potential job displacement remain significant challenges that researchers and policymakers are working to address.",
            "summary": "AI is advancing rapidly with sophisticated machine learning models, while researchers address concerns about safety, ethics, and job displacement.",
        },
        {
            "text": "Climate change is causing more frequent and severe weather events around the world. Rising global temperatures are leading to melting ice caps, rising sea levels, and changing precipitation patterns. Scientists warn that without significant reductions in greenhouse gas emissions, these trends will continue to worsen, potentially leading to catastrophic consequences for ecosystems and human societies.",
            "summary": "Climate change is causing severe weather events, melting ice caps, and rising sea levels, with scientists warning of catastrophic consequences without emissions reductions.",
        },
    ]
    
    while len(samples) < num_samples:
        samples.extend(samples[:num_samples - len(samples)])
    
    return samples[:num_samples]

def _load_wmt16_samples(num_samples: int) -> List[Dict[str, Any]]:
    
    samples = [
        {
            "input": "The weather is beautiful today.",
            "output": "Le temps est magnifique aujourd'hui.",
        },
        {
            "input": "I would like to order a coffee, please.",
            "output": "Je voudrais commander un café, s'il vous plaît.",
        },
        {
            "input": "Where is the nearest train station?",
            "output": "Où se trouve la gare la plus proche?",
        },
    ]
    
    while len(samples) < num_samples:
        samples.extend(samples[:num_samples - len(samples)])
    
    return samples[:num_samples]

def _load_glue_samples(num_samples: int) -> List[Dict[str, Any]]:
    
    samples = [
        {
            "input": "This movie was fantastic! I loved every minute of it.",
            "output": "positive",
        },
        {
            "input": "The service was terrible and the food was cold.",
            "output": "negative",
        },
        {
            "input": "The product arrived on time but was slightly damaged.",
            "output": "neutral",
        },
        {
            "input": "I can't believe how amazing this concert was!",
            "output": "positive",
        },
        {
            "input": "This book was a complete waste of time and money.",
            "output": "negative",
        },
    ]
    
    while len(samples) < num_samples:
        samples.extend(samples[:num_samples - len(samples)])
    
    return samples[:num_samples]

def _load_generic_samples(num_samples: int) -> List[Dict[str, Any]]:
    
    samples = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data.",
            "question": "What is machine learning?",
            "answer": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data.",
            "text": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data.",
            "summary": "Machine learning is an AI field focused on data-driven learning systems.",
        },
        {
            "input": "Explain quantum computing.",
            "output": "Quantum computing uses quantum bits or qubits to perform computations that would be much more difficult for classical computers.",
            "question": "Explain quantum computing.",
            "answer": "Quantum computing uses quantum bits or qubits to perform computations that would be much more difficult for classical computers.",
            "text": "Quantum computing uses quantum bits or qubits to perform computations that would be much more difficult for classical computers.",
            "summary": "Quantum computing uses qubits for computations difficult for classical computers.",
        },
    ]
    
    while len(samples) < num_samples:
        samples.extend(samples[:num_samples - len(samples)])
    
    return samples[:num_samples]
