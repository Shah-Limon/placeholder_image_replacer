#!/usr/bin/env python3
# .github/scripts/placeholder_image_replacer.py

import os
import re
import time
import requests
import yaml
import argparse
from PIL import Image
from google import genai
from google.genai import types
import mimetypes
import glob
import logging
from datetime import datetime
import sys

# Setup argument parser
parser = argparse.ArgumentParser(description='Replace placeholder images in markdown files')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()

# Configure logging
log_level = logging.INFO if args.verbose else logging.WARNING
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
PLACEHOLDER_IMAGE_URL = "https://res.cloudinary.com/dbcpfy04c/image/upload/v1743184673/images_k6zam3.png"
GENERATED_ARTICLES_DIR = "generated-articles"
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUMMARY_FILE = "image_replacement_summary.txt"

# Rate limit configuration
MAX_RETRIES = 5        # Maximum number of retries for API calls
BASE_RETRY_DELAY = 5   # Base delay in seconds before retrying
MAX_DELAY = 120        # Maximum delay in seconds
SUCCESS_DELAY = 60     # Wait 1 minute (60 seconds) after successful image creation

# Summary statistics
stats = {
    "total_files": 0,
    "files_with_placeholders": 0,
    "successful_replacements": 0,
    "failed_replacements": 0,
    "skipped_files": 0,
    "replaced_images": []
}

def save_binary_file(file_name, data):
    """Save binary data to a file"""
    with open(file_name, "wb") as f:
        f.write(data)
    logger.info(f"Saved binary data to {file_name}")

def compress_image(image_path, quality=85):
    """Compress and convert image to WebP format"""
    try:
        with Image.open(image_path) as img:
            webp_path = f"{os.path.splitext(image_path)[0]}.webp"
            img.save(webp_path, 'WEBP', quality=quality)
            os.remove(image_path)  # Remove original file
            logger.info(f"Compressed image and saved as {webp_path}")
            return webp_path
    except Exception as e:
        logger.error(f"Image compression error: {e}")
        return image_path

def upload_to_cloudinary(file_path, resource_type="image"):
    """Upload file to Cloudinary
    
    Raises appropriate exceptions on rate limit or other errors
    for proper handling by the calling function
    """
    logger.info(f"Uploading {file_path} to Cloudinary...")
    url = f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/{resource_type}/upload"
    payload = {
        'upload_preset': 'ml_default',
        'api_key': CLOUDINARY_API_KEY
    }
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, data=payload, files=files)
    
    if response.status_code == 200:
        image_url = response.json()['secure_url']
        logger.info(f"✅ Successfully uploaded to Cloudinary: {image_url}")
        return image_url
    elif response.status_code == 429:
        # Rate limit error
        logger.warning(f"⚠️ Cloudinary rate limit exceeded: {response.text}")
        raise Exception(f"Cloudinary rate limit exceeded: {response.text}")
    elif response.status_code >= 500:
        # Server error
        logger.error(f"❌ Cloudinary server error: {response.text}")
        raise Exception(f"Cloudinary server error: {response.text}")
    else:
        # Other error
        logger.error(f"❌ Cloudinary upload failed with status {response.status_code}: {response.text}")
        raise Exception(f"Cloudinary upload failed with status {response.status_code}: {response.text}")

def generate_and_upload_image(title, max_retries=3, retry_delay=5):
    """Generate an image with Gemini API and upload to Cloudinary
    
    Args:
        title: The article title to generate an image for
        max_retries: Maximum number of retries for API calls (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 5)
    """
    print(f"\n🎨 Generating image for: '{title}'")
    logger.info(f"Starting image generation process for: {title}")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  ↳ Attempt {attempt}/{max_retries} using Gemini API")
            logger.info(f"Image generation attempt {attempt}/{max_retries}")
            
            client = genai.Client(api_key=GEMINI_API_KEY)
            model = "gemini-2.0-flash-exp-image-generation"
            contents = [types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Create a realistic blog header image for: {title}")]
            )]
            
            print(f"  ↳ Requesting image from Gemini API...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["image", "text"])
            )

            if (response.candidates and response.candidates[0].content.parts and 
                len(response.candidates[0].content.parts) > 0):
                
                print(f"  ↳ Received response with {len(response.candidates[0].content.parts)} parts")
                logger.info(f"Received response with {len(response.candidates[0].content.parts)} parts")
                
                # Get the first part and verify it has inline_data
                first_part = response.candidates[0].content.parts[0]
                
                if not hasattr(first_part, 'inline_data') or first_part.inline_data is None:
                    print(f"  ↳ Response does not contain image data")
                    logger.error(f"Response part does not contain inline_data attribute")
                    raise Exception("Response does not contain valid image data")
                
                inline_data = first_part.inline_data
                
                # Verify mime_type exists
                if not hasattr(inline_data, 'mime_type') or inline_data.mime_type is None:
                    print(f"  ↳ Response inline_data does not have mime_type")
                    logger.error(f"Response inline_data does not have mime_type attribute")
                    raise Exception("Response inline_data missing mime_type")
                
                # Verify data exists
                if not hasattr(inline_data, 'data') or inline_data.data is None:
                    print(f"  ↳ Response inline_data does not have data")
                    logger.error(f"Response inline_data does not have data attribute")
                    raise Exception("Response inline_data missing data")
                
                print(f"  ↳ Image received from API (mime_type: {inline_data.mime_type})")
                logger.info(f"Successfully received image from Gemini API (mime_type: {inline_data.mime_type})")
                
                file_ext = mimetypes.guess_extension(inline_data.mime_type)
                original_file = f"generated_image_{int(time.time())}{file_ext}"
                save_binary_file(original_file, inline_data.data)

                # Compress and convert to WebP
                print(f"  ↳ Compressing image...")
                final_file = compress_image(original_file)
                
                # Upload to Cloudinary with retries
                image_url = None
                for upload_attempt in range(1, max_retries + 1):
                    try:
                        print(f"  ↳ Uploading to Cloudinary (attempt {upload_attempt}/{max_retries})...")
                        logger.info(f"Upload attempt {upload_attempt}/{max_retries}")
                        image_url = upload_to_cloudinary(final_file)
                        if image_url:
                            break
                        current_delay = retry_delay * (2 ** (upload_attempt - 1))  # Exponential backoff
                        print(f"  ↳ Upload failed, retrying in {current_delay}s...")
                        logger.warning(f"Upload failed, retrying in {current_delay} seconds")
                        time.sleep(current_delay)
                    except Exception as e:
                        print(f"  ↳ Upload error: {e}")
                        logger.error(f"Upload error (attempt {upload_attempt}/{max_retries}): {e}")
                        if "rate limit" in str(e).lower():
                            current_delay = retry_delay * (2 ** (upload_attempt - 1))  # Exponential backoff
                            print(f"  ↳ Rate limit hit, waiting {current_delay}s before retry...")
                            logger.warning(f"Rate limit hit, waiting {current_delay} seconds before retry")
                            time.sleep(current_delay)
                
                # Clean up local file
                if os.path.exists(final_file):
                    os.remove(final_file)
                    logger.info(f"Removed temporary file {final_file}")
                
                if image_url:
                    print(f"  ↳ ✅ Image successfully generated and uploaded")
                    return image_url
            else:
                print(f"  ↳ Response did not contain any valid parts")
                logger.error(f"Response did not contain any valid parts")
            
            # If we got here, something failed but didn't raise an exception
            # Wait with exponential backoff before retrying
            if attempt < max_retries:
                current_delay = retry_delay * (2 ** (attempt - 1))
                print(f"  ↳ Generation failed, retrying in {current_delay}s...")
                logger.warning(f"Generation failed, retrying in {current_delay} seconds")
                time.sleep(current_delay)
            
        except Exception as e:
            error_message = str(e).lower()
            print(f"  ↳ ❌ Error: {e}")
            logger.error(f"Image generation error (attempt {attempt}/{max_retries}): {e}")
            
            # Check for rate limit errors
            if "rate limit" in error_message or "quota" in error_message or "429" in error_message:
                if attempt < max_retries:
                    current_delay = retry_delay * (2 ** attempt)  # Longer exponential backoff for rate limits
                    print(f"  ↳ API rate limit hit, waiting {current_delay}s before retry...")
                    logger.warning(f"API rate limit hit, waiting {current_delay} seconds before retry")
                    time.sleep(current_delay)
            elif attempt < max_retries:
                current_delay = retry_delay * (2 ** (attempt - 1))
                print(f"  ↳ Error occurred, retrying in {current_delay}s...")
                logger.warning(f"Error occurred, retrying in {current_delay} seconds")
                time.sleep(current_delay)
    
    print(f"  ↳ ❌ All attempts failed for generating image for '{title}'")
    logger.error(f"All attempts failed for generating image for '{title}'")
    return None

# Replace the entire extract_front_matter function with this improved version
def extract_front_matter(content):
    """Extract front matter from markdown content with better error handling"""
    print(f"  ↳ Attempting to extract front matter...")
    logger.info(f"Attempting to extract front matter")
    
    # Print first few characters of content for debugging
    content_preview = content[:100].replace('\n', '\\n')
    logger.debug(f"Content preview: {content_preview}...")
    
    # Check if content starts with --- which indicates front matter
    if not content.startswith('---'):
        print(f"  ↳ Content does not start with '---', no valid front matter")
        logger.error("Content does not start with '---', no valid front matter")
        return None, None
    
    # Find the second --- that closes the front matter block
    try:
        # Find the second occurrence of ---
        first_marker_end = content.find('---', 0) + 3
        second_marker_start = content.find('---', first_marker_end)
        
        if second_marker_start == -1:
            print(f"  ↳ Could not find closing '---' for front matter")
            logger.error("Could not find closing '---' for front matter")
            return None, None
        
        # Extract the text between the markers
        front_matter_text = content[first_marker_end:second_marker_start].strip()
        full_front_matter = content[:second_marker_start+3]
        
        try:
            front_matter = yaml.safe_load(front_matter_text)
            if front_matter and isinstance(front_matter, dict):
                # Explicitly check for title
                if 'title' in front_matter:
                    print(f"  ↳ Successfully extracted front matter with title: '{front_matter['title']}'")
                    logger.info(f"Successfully extracted front matter with title: '{front_matter['title']}'")
                else:
                    print(f"  ↳ Front matter found but no 'title' field: {list(front_matter.keys())}")
                    logger.warning(f"Front matter found but no 'title' field. Available keys: {list(front_matter.keys())}")
                
                return front_matter, full_front_matter
            else:
                print(f"  ↳ Front matter parsed but returned {type(front_matter)} instead of a dictionary")
                logger.error(f"Front matter parsed but returned {type(front_matter)} instead of a dictionary")
        except yaml.YAMLError as e:
            print(f"  ↳ YAML parsing error: {e}")
            logger.error(f"YAML parsing error: {e}")
            print(f"  ↳ Front matter text: {front_matter_text[:100]}...")
            logger.error(f"Front matter text: {front_matter_text[:100]}...")
    except Exception as e:
        print(f"  ↳ Unexpected error extracting front matter: {e}")
        logger.error(f"Unexpected error extracting front matter: {e}")
    
    return None, None

# Also update the replace_image_in_markdown function to add more debugging info
def replace_image_in_markdown(md_file_path):
    """Check and replace placeholder image in markdown file with improved error handling"""
    try:
        file_basename = os.path.basename(md_file_path)
        print(f"\n📄 Processing file: {file_basename}")
        logger.info(f"Processing file: {md_file_path}")
        
        # Check if file exists and is readable
        if not os.path.exists(md_file_path):
            print(f"  ↳ ❌ File does not exist: {md_file_path}")
            logger.error(f"File does not exist: {md_file_path}")
            stats["failed_replacements"] += 1
            return False
            
        # Read the file with explicit UTF-8 encoding
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content_length = len(content)
                print(f"  ↳ Successfully read file ({content_length} bytes)")
                logger.info(f"Successfully read file ({content_length} bytes)")
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            with open(md_file_path, 'r', encoding='latin-1') as f:
                content = f.read()
                content_length = len(content)
                print(f"  ↳ Read file with latin-1 encoding ({content_length} bytes)")
                logger.info(f"Read file with latin-1 encoding ({content_length} bytes)")
        
        # Check if placeholder image exists in content
        if PLACEHOLDER_IMAGE_URL not in content:
            print(f"  ↳ No placeholder image found")
            logger.info(f"No placeholder image found in {md_file_path}")
            stats["skipped_files"] += 1
            return False
        
        stats["files_with_placeholders"] += 1
        
        # Extract front matter to get title
        front_matter, front_matter_text = extract_front_matter(content)
        if not front_matter or 'title' not in front_matter:
            print(f"  ↳ Could not extract title from file")
            logger.error(f"Could not extract title from {md_file_path}")
            
            # Additional debugging - try to find title directly with regex
            title_match = re.search(r'^title:\s*(.+?)$', content, re.MULTILINE)
            if title_match:
                extracted_title = title_match.group(1).strip()
                print(f"  ↳ Alternative title extraction found: '{extracted_title}'")
                logger.info(f"Alternative title extraction found: '{extracted_title}'")
                title = extracted_title
            else:
                stats["failed_replacements"] += 1
                return False
        else:
            title = front_matter['title']
        
        # Generate and upload new image with retries
        new_image_url = generate_and_upload_image(title, max_retries=MAX_RETRIES, retry_delay=BASE_RETRY_DELAY)
        if not new_image_url:
            print(f"  ↳ Failed to generate image after {MAX_RETRIES} attempts")
            logger.error(f"Failed to generate image for '{title}' after {MAX_RETRIES} attempts")
            stats["failed_replacements"] += 1
            return False
        
        # Replace placeholder with new image URL
        updated_content = content.replace(PLACEHOLDER_IMAGE_URL, new_image_url)
        
        # Write updated content back to file
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"  ↳ ✅ Successfully replaced placeholder image")
        print(f"  ↳ New image URL: {new_image_url}")
        logger.info(f"Successfully replaced placeholder image in {file_basename}")
        logger.info(f"New image URL: {new_image_url}")
        
        # Create a backup of the original file
        backup_path = f"{md_file_path}.bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Backup saved to {backup_path}")
        
        stats["successful_replacements"] += 1
        stats["replaced_images"].append({
            "file": file_basename,
            "title": title,
            "image_url": new_image_url
        })
        
        return True
    
    except Exception as e:
        print(f"  ↳ ❌ Error: {e}")
        logger.error(f"Error processing {os.path.basename(md_file_path)}: {e}")
        stats["failed_replacements"] += 1
        # Propagate the exception for rate limit handling
        raise
def process_all_markdown_files():
    """Process all markdown files in the generated articles directory with rate limit handling"""
    # Get all markdown files
    md_files = glob.glob(os.path.join(GENERATED_ARTICLES_DIR, "*.md"))
    
    if not md_files:
        print(f"⚠️ No markdown files found in {GENERATED_ARTICLES_DIR}")
        logger.warning(f"No markdown files found in {GENERATED_ARTICLES_DIR}")
        return
    
    stats["total_files"] = len(md_files)
    print(f"🔍 Found {len(md_files)} markdown files to process")
    logger.info(f"Found {len(md_files)} markdown files to process")
    
    # Base delay between files (will increase if rate limits are hit)
    base_delay = 3  # seconds
    current_delay = base_delay
    max_delay = 60  # Maximum delay in seconds
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    for i, md_file in enumerate(md_files):
        file_basename = os.path.basename(md_file)
        print(f"\n📝 File {i+1}/{len(md_files)}: {file_basename}")
        logger.info(f"Processing ({i+1}/{len(md_files)}): {file_basename}")
        
        try:
            if replace_image_in_markdown(md_file):
                # Reset delay and failure counter after success
                consecutive_failures = 0
                
                # Wait for 1 minute (60 seconds) after successful image creation
                print(f"⏱️ Waiting {SUCCESS_DELAY}s before next file...")
                logger.info(f"Success! Waiting {SUCCESS_DELAY}s before next file as requested")
                time.sleep(SUCCESS_DELAY)
                
                # Reset current delay to base delay for next attempt
                current_delay = base_delay
            else:
                consecutive_failures += 1
                # Use standard delay between failed files
                print(f"⏱️ Waiting {current_delay}s before next file...")
                logger.info(f"Waiting {current_delay}s before next file")
                time.sleep(current_delay)
        except Exception as e:
            consecutive_failures += 1
            
            # Check if it's likely a rate limit issue
            error_message = str(e).lower()
            if "rate limit" in error_message or "quota" in error_message or "429" in error_message:
                # Increase delay with rate limit errors
                current_delay = min(current_delay * 2, max_delay)
                print(f"⚠️ Rate limit detected. Increasing delay to {current_delay}s")
                logger.warning(f"Rate limit detected. Increasing delay to {current_delay}s")
            else:
                logger.error(f"Error processing {file_basename}: {e}")
            
            # Wait before trying next file
            print(f"⏱️ Waiting {current_delay}s before next file...")
            logger.info(f"Waiting {current_delay}s before next file")
            time.sleep(current_delay)
        
        # Pause processing if we have too many consecutive failures
        if consecutive_failures >= max_consecutive_failures:
            extended_break = min(current_delay * 3, 120)
            print(f"⚠️ Too many consecutive failures ({consecutive_failures}). Taking a {extended_break}s break...")
            logger.warning(f"Too many consecutive failures ({consecutive_failures}). Taking a longer break of {extended_break}s")
            time.sleep(extended_break)  # Extended break
            consecutive_failures = 0
    
    # Create summary file
    create_summary_file()

def create_summary_file():
    """Create a summary file of the image replacement process"""
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("# Image Replacement Summary\n\n")
        f.write(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Statistics\n")
        f.write(f"- Total files scanned: {stats['total_files']}\n")
        f.write(f"- Files with placeholder images: {stats['files_with_placeholders']}\n")
        f.write(f"- Successful replacements: {stats['successful_replacements']}\n")
        f.write(f"- Failed replacements: {stats['failed_replacements']}\n")
        f.write(f"- Skipped files (no placeholders): {stats['skipped_files']}\n\n")
        
        f.write("## Replaced Images\n")
        if stats["replaced_images"]:
            for i, item in enumerate(stats["replaced_images"], 1):
                f.write(f"### {i}. {item['file']}\n")
                f.write(f"- Title: {item['title']}\n")
                f.write(f"- Image URL: {item['image_url']}\n")
                f.write(f"- Preview: ![{item['title']}]({item['image_url']})\n\n")
        else:
            f.write("No images were replaced in this run.\n")
    
    logger.info(f"Created summary file: {SUMMARY_FILE}")
    print(f"\n📋 Created summary file: {SUMMARY_FILE}")

def main():
    print(f"🚀 Starting placeholder image replacement script - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ Settings: Max retries: {MAX_RETRIES}, Base delay: {BASE_RETRY_DELAY}s, Success delay: {SUCCESS_DELAY}s")
    logger.info(f"Starting placeholder image replacement script")
    logger.info(f"Max retries: {MAX_RETRIES}, Base delay: {BASE_RETRY_DELAY}s, Max delay: {MAX_DELAY}s")
    
    # Check environment variables
    if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, GEMINI_API_KEY]):
        print("❌ Error: Required environment variables are not set")
        print(f"CLOUDINARY_CLOUD_NAME: {'✅ Set' if CLOUDINARY_CLOUD_NAME else '❌ Not Set'}")
        print(f"CLOUDINARY_API_KEY: {'✅ Set' if CLOUDINARY_API_KEY else '❌ Not Set'}")
        print(f"GEMINI_API_KEY: {'✅ Set' if GEMINI_API_KEY else '❌ Not Set'}")
        logger.error("Required environment variables are not set")
        return
    
    try:
        # Process all markdown files
        process_all_markdown_files()
        print(f"\n✅ Script completed successfully - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Summary: Processed {stats['total_files']} files, replaced {stats['successful_replacements']} images")
        logger.info(f"Script completed successfully")
        logger.info(f"Summary: Processed {stats['total_files']} files, replaced {stats['successful_replacements']} images")
    except KeyboardInterrupt:
        print("\n⚠️ Script interrupted by user. Exiting...")
        logger.warning("Script interrupted by user")
        # Create summary file even if interrupted
        create_summary_file()
    except Exception as e:
        print(f"\n❌ Unexpected error in main execution: {e}")
        logger.error(f"Unexpected error in main execution: {e}")
        print("Script terminated with errors.")
        # Create summary file even if errors occurred
        create_summary_file()

if __name__ == "__main__":
    main()