#!/usr/bin/env python3
# .github/scripts/placeholder_image_replacer.py

import os
import re
import time
import requests
import yaml
from PIL import Image
from google import genai
from google.genai import types
import mimetypes
import glob
import logging
from datetime import datetime
import sys

# Constants
PLACEHOLDER_IMAGE_URL = "https://res.cloudinary.com/dbcpfy04c/image/upload/v1743184673/images_k6zam3.png"
GENERATED_ARTICLES_DIR = "generated-articles"
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Rate limit configuration
MAX_RETRIES = 5        # Maximum number of retries for API calls
BASE_RETRY_DELAY = 5   # Base delay in seconds before retrying
MAX_DELAY = 120        # Maximum delay in seconds
SUCCESS_DELAY = 60     # Wait 1 minute (60 seconds) after successful image creation

def save_binary_file(file_name, data):
    """Save binary data to a file"""
    with open(file_name, "wb") as f:
        f.write(data)

def compress_image(image_path, quality=85):
    """Compress and convert image to WebP format"""
    try:
        with Image.open(image_path) as img:
            webp_path = f"{os.path.splitext(image_path)[0]}.webp"
            img.save(webp_path, 'WEBP', quality=quality)
            os.remove(image_path)  # Remove original file
            return webp_path
    except Exception as e:
        print(f"Image compression error: {e}")
        return image_path

def upload_to_cloudinary(file_path, resource_type="image"):
    """Upload file to Cloudinary
    
    Raises appropriate exceptions on rate limit or other errors
    for proper handling by the calling function
    """
    url = f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/{resource_type}/upload"
    payload = {
        'upload_preset': 'ml_default',
        'api_key': CLOUDINARY_API_KEY
    }
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, data=payload, files=files)
    
    if response.status_code == 200:
        return response.json()['secure_url']
    elif response.status_code == 429:
        # Rate limit error
        raise Exception(f"Cloudinary rate limit exceeded: {response.text}")
    elif response.status_code >= 500:
        # Server error
        raise Exception(f"Cloudinary server error: {response.text}")
    else:
        # Other error
        raise Exception(f"Cloudinary upload failed with status {response.status_code}: {response.text}")

def generate_and_upload_image(title, max_retries=3, retry_delay=5):
    """Generate an image with Gemini API and upload to Cloudinary
    
    Args:
        title: The article title to generate an image for
        max_retries: Maximum number of retries for API calls (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 5)
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Image generation attempt {attempt}/{max_retries} for: {title}")
            client = genai.Client(api_key=GEMINI_API_KEY)
            model = "gemini-2.0-flash-exp-image-generation"
            contents = [types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Create a realistic blog header image for: {title}")]
            )]
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["image", "text"])
            )

            if response.candidates and response.candidates[0].content.parts:
                inline_data = response.candidates[0].content.parts[0].inline_data
                file_ext = mimetypes.guess_extension(inline_data.mime_type)
                original_file = f"generated_image_{int(time.time())}{file_ext}"
                save_binary_file(original_file, inline_data.data)

                # Compress and convert to WebP
                final_file = compress_image(original_file)
                
                # Upload to Cloudinary with retries
                image_url = None
                for upload_attempt in range(1, max_retries + 1):
                    try:
                        print(f"Upload attempt {upload_attempt}/{max_retries}")
                        image_url = upload_to_cloudinary(final_file)
                        if image_url:
                            break
                        current_delay = retry_delay * (2 ** (upload_attempt - 1))  # Exponential backoff
                        print(f"Upload failed, retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                    except Exception as e:
                        print(f"Upload error (attempt {upload_attempt}/{max_retries}): {e}")
                        if "rate limit" in str(e).lower():
                            current_delay = retry_delay * (2 ** (upload_attempt - 1))  # Exponential backoff
                            print(f"Rate limit hit, waiting {current_delay} seconds before retry...")
                            time.sleep(current_delay)
                
                # Clean up local file
                if os.path.exists(final_file):
                    os.remove(final_file)
                
                if image_url:
                    return image_url
            
            # If we got here, something failed but didn't raise an exception
            # Wait with exponential backoff before retrying
            if attempt < max_retries:
                current_delay = retry_delay * (2 ** (attempt - 1))
                print(f"Generation failed, retrying in {current_delay} seconds...")
                time.sleep(current_delay)
            
        except Exception as e:
            error_message = str(e).lower()
            print(f"Image generation error (attempt {attempt}/{max_retries}): {e}")
            
            # Check for rate limit errors
            if "rate limit" in error_message or "quota" in error_message or "429" in error_message:
                if attempt < max_retries:
                    current_delay = retry_delay * (2 ** attempt)  # Longer exponential backoff for rate limits
                    print(f"API rate limit hit, waiting {current_delay} seconds before retry...")
                    time.sleep(current_delay)
            elif attempt < max_retries:
                current_delay = retry_delay * (2 ** (attempt - 1))
                print(f"Error occurred, retrying in {current_delay} seconds...")
                time.sleep(current_delay)
    
    print(f"All attempts failed for generating image for '{title}'")
    return None

def extract_front_matter(content):
    """Extract front matter from markdown content"""
    pattern = r'^---\s*(.*?)\s*---\s*'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        front_matter_text = match.group(1)
        try:
            front_matter = yaml.safe_load(front_matter_text)
            return front_matter, match.group(0)
        except yaml.YAMLError as e:
            print(f"Error parsing front matter: {e}")
    return None, None

def replace_image_in_markdown(md_file_path):
    """Check and replace placeholder image in markdown file"""
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if placeholder image exists in content
        if PLACEHOLDER_IMAGE_URL not in content:
            print(f"No placeholder image found in {md_file_path}")
            return False
        
        # Extract front matter to get title
        front_matter, _ = extract_front_matter(content)
        if not front_matter or 'title' not in front_matter:
            print(f"Could not extract title from {md_file_path}")
            return False
        
        title = front_matter['title']
        print(f"Generating new image for: {title}")
        
        # Generate and upload new image with retries
        new_image_url = generate_and_upload_image(title, max_retries=MAX_RETRIES, retry_delay=BASE_RETRY_DELAY)
        if not new_image_url:
            print(f"Failed to generate image for '{title}' after {MAX_RETRIES} attempts")
            return False
        
        # Replace placeholder with new image URL
        updated_content = content.replace(PLACEHOLDER_IMAGE_URL, new_image_url)
        
        # Write updated content back to file
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Successfully replaced placeholder image in {os.path.basename(md_file_path)}")
        print(f"   New image URL: {new_image_url}")
        
        # Create a backup of the original file
        backup_path = f"{md_file_path}.bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   Backup saved to {os.path.basename(backup_path)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(md_file_path)}: {e}")
        # Propagate the exception for rate limit handling
        raise

def process_all_markdown_files():
    """Process all markdown files in the generated articles directory with rate limit handling"""
    # Get all markdown files
    md_files = glob.glob(os.path.join(GENERATED_ARTICLES_DIR, "*.md"))
    
    if not md_files:
        print(f"No markdown files found in {GENERATED_ARTICLES_DIR}")
        return
    
    print(f"Found {len(md_files)} markdown files to process")
    
    success_count = 0
    failed_files = []
    
    # Base delay between files (will increase if rate limits are hit)
    base_delay = 3  # seconds
    current_delay = base_delay
    max_delay = 60  # Maximum delay in seconds
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    for i, md_file in enumerate(md_files):
        file_basename = os.path.basename(md_file)
        print(f"\nProcessing ({i+1}/{len(md_files)}): {file_basename}")
        
        try:
            if replace_image_in_markdown(md_file):
                success_count += 1
                # Reset delay and failure counter after success
                consecutive_failures = 0
                
                # Wait for 1 minute (60 seconds) after successful image creation
                print(f"Success! Waiting {SUCCESS_DELAY}s before next file as requested...")
                time.sleep(SUCCESS_DELAY)
                
                # Reset current delay to base delay for next attempt
                current_delay = base_delay
            else:
                failed_files.append(file_basename)
                consecutive_failures += 1
                print(f"Failed to process {file_basename}")
                # Use standard delay between failed files
                print(f"Waiting {current_delay}s before next file...")
                time.sleep(current_delay)
        except Exception as e:
            failed_files.append(file_basename)
            consecutive_failures += 1
            
            # Check if it's likely a rate limit issue
            error_message = str(e).lower()
            if "rate limit" in error_message or "quota" in error_message or "429" in error_message:
                # Increase delay with rate limit errors
                current_delay = min(current_delay * 2, max_delay)
                print(f"Rate limit detected. Increasing delay to {current_delay}s")
            else:
                print(f"Error processing {file_basename}: {e}")
            
            # Wait before trying next file
            print(f"Waiting {current_delay}s before next file...")
            time.sleep(current_delay)
        
        # Pause processing if we have too many consecutive failures
        if consecutive_failures >= max_consecutive_failures:
            print(f"Too many consecutive failures ({consecutive_failures}). Taking a longer break...")
            time.sleep(min(current_delay * 3, 120))  # Extended break
            consecutive_failures = 0
    
    print(f"\nProcessing complete. Replaced images in {success_count} out of {len(md_files)} files.")
    
    if failed_files:
        print("\nFailed files:")
        for i, failed_file in enumerate(failed_files, 1):
            print(f"{i}. {failed_file}")
        print("\nYou may want to retry these files after waiting a while.")

def main():
    print(f"Starting placeholder image replacement script - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max retries: {MAX_RETRIES}, Base delay: {BASE_RETRY_DELAY}s, Max delay: {MAX_DELAY}s")
    print(f"Success delay: {SUCCESS_DELAY}s (waiting 1 minute after each successful image creation)")
    
    # Check environment variables
    if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, GEMINI_API_KEY]):
        print("Error: Required environment variables are not set")
        print(f"CLOUDINARY_CLOUD_NAME: {'Set' if CLOUDINARY_CLOUD_NAME else 'Not Set'}")
        print(f"CLOUDINARY_API_KEY: {'Set' if CLOUDINARY_API_KEY else 'Not Set'}")
        print(f"GEMINI_API_KEY: {'Set' if GEMINI_API_KEY else 'Not Set'}")
        return
    
    try:
        # Process all markdown files
        process_all_markdown_files()
        print(f"Script completed successfully - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")
        print("Script terminated with errors.")

if __name__ == "__main__":
    main()