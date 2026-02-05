"""
Vision Tool - Tools for CLIP embeddings and VLM analysis
Handles image processing and visual understanding tasks
"""

#the data cleaning will be done by the scientist agent

import os
from PIL import Image
from typing import Dict

def image_profiler(folder_path: str) -> Dict:
    """
    Analyzes a directory of images to provide structural metadata.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        return {"error": "No valid images found in path"}

    # Analyze the first few images to get a "sense" of the data
    sample_data = []
    formats = set()
    modes = set() # RGB, CMYK, etc.
    
    for img_name in image_files[:5]:
        with Image.open(os.path.join(folder_path, img_name)) as img:
            sample_data.append(img.size) # (width, height)
            formats.add(img.format)
            modes.add(img.mode)

    return {
        "total_images": len(image_files),
        "formats": list(formats),
        "color_modes": list(modes),
        "sample_resolutions": sample_data,
        "average_res": [sum(x)/len(x) for x in zip(*sample_data)],
        "is_uniform_size": len(set(sample_data)) == 1
    }