from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import cv2
from PIL import Image, ImageOps
import numpy as np
import re
from typing import Union, List
import base64
from pathlib import Path
import pickle
from io import BytesIO
import uuid

import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Month name to number mapping
MONTHS = {
    "Jan": "1", "Feb": "2", "Mar": "3", "Apr": "4", "May": "5", "Jun": "6",
    "Jul": "7", "Aug": "8", "Sep": "9", "Oct": "10", "Nov": "11", "Dec": "12"
}

def build_transform(input_size):
    """
    Build image transformation pipeline for model input
    Args:
        input_size: Target size for image transformation
    Returns:
        Composed transformation pipeline
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find the closest matching aspect ratio from target ratios
    Args:
        aspect_ratio: Current image aspect ratio
        target_ratios: List of possible target ratios
        width: Image width
        height: Image height
        image_size: Target image size
    Returns:
        Best matching ratio tuple
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess image for model input by splitting into blocks
    Args:
        image: Input image
        min_num: Minimum number of blocks
        max_num: Maximum number of blocks
        image_size: Target size for each block
        use_thumbnail: Whether to include thumbnail
    Returns:
        List of processed image blocks
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate possible target ratios
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find best ratio and calculate dimensions
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Process image into blocks
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images

def straighten_images(images: Union[List[str], List[np.ndarray], List[Image.Image]]) -> List[Image.Image]:
    """
    Straighten images based on text orientation detection
    Args:
        images: List of images in various formats
    Returns:
        List of straightened PIL Images
    """
    straightened_images = []

    for img in images:
        # Convert input to numpy array
        if isinstance(img, str):
            image = cv2.imread(img)
        elif isinstance(img, Image.Image):
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            image = img.copy()
        else:
            raise ValueError("Unsupported image type")

        # Scale image for better processing
        scale_percent = 300
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        try:
            # Detect and correct orientation
            cv2.imwrite(f"resized_image_{datetime.datetime.now()}.jpg", resized_image)
            osd = pytesseract.image_to_osd(resized_image)
            rotation_angle = int(re.search(r"Rotate: (\d+)", osd).group(1))

            if rotation_angle != 0:
                # Calculate rotation parameters
                (h, w) = resized_image.shape[:2]
                center = (w // 2, h // 2)
                angle = 90 if rotation_angle == 90 else -90 if rotation_angle == 270 else rotation_angle
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                # Adjust dimensions for rotation
                cos = np.abs(matrix[0, 0])
                sin = np.abs(matrix[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                matrix[0, 2] += (new_w / 2) - center[0]
                matrix[1, 2] += (new_h / 2) - center[1]

                # Perform rotation
                rotated_image = cv2.warpAffine(
                    resized_image, matrix, (new_w, new_h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
                straightened_images.append(Image.fromarray(rotated_image))
            else:
                straightened_images.append(Image.fromarray(resized_image))

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            straightened_images.append(resized_image)

    return straightened_images

def save_image_to_tmp_dir(image: Image.Image, file_extension: str = "png") -> str:
    """
    Save PIL image to temporary directory
    Args:
        image: PIL Image to save
        file_extension: File extension for saved image
    Returns:
        Path to saved image
    """
    # image_path = Path("./tmp") / f"image.{file_extension}"
    # image.save(image_path)
    # return str(image_path)

    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    image_path = Path("./tmp") / unique_filename
    image.save(image_path)
    return str(image_path)


def convert_image_to_base64(image: Image.Image, file_extension: str = "png") -> str:
    """
    Convert PIL image to base64 string
    Args:
        image: PIL Image to convert
        file_extension: Image format extension
    Returns:
        Base64 encoded image string with data URI prefix
    """
    buffer = BytesIO()
    image.save(buffer, format=file_extension.upper())
    buffer.seek(0)
    encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
    return "data:image/png;base64," + encoded_string

def make_image_square(img: Image.Image):
    """
    Convert image to square by adding white padding
    Args:
        img: Input PIL Image
    Returns:
        Square PIL Image with white padding
    """
    width, height = img.size
    square_size = max(width, height)
    square_img = Image.new("RGB", (square_size, square_size), (255, 255, 255))
    paste_x = (square_size - width) // 2
    paste_y = (square_size - height) // 2
    square_img.paste(img, (paste_x, paste_y))
    return square_img

def base64topil(img_str):
    """
    Convert base64 image string to PIL Image
    Args:
        img_str: Base64 encoded image string
    Returns:
        PIL Image or None if conversion fails
    """
    try:
        MAX_PIXELS = Image.MAX_IMAGE_PIXELS or 178956970
        img = Image.open(BytesIO(base64.b64decode(img_str.split(",")[1])))
        return resize_large_image(img, MAX_PIXELS)

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# Month name mappings
MONTHS = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

def preprocess_date(date_str):
    """
    Normalize date string to MM/DD/YY or MM/DD format
    Handles formats: 02-09-24, 02/09/24, Oct 24 2024, Oct-24-2024, 02.09.24, 02/09
    """
    try:
        # Handle null/empty cases
        if not date_str or date_str in ["null", "empty", None, "", "N/A"]:
            return "null"

        # Clean input - preserve hyphens for now
        date_str = str(date_str).strip()

        # Remove time components (e.g., "10/31/25 07:00" -> "10/31/25")
        # This handles HH:MM, H:MM, and am/pm formats
        date_str = re.sub(r'\s+\d{1,2}:\d{2}.*$', '', date_str)  # Remove time with colon
        date_str = re.sub(r'\s+\d{1,2}(am|pm)?$', '', date_str, flags=re.IGNORECASE)  # Remove hour only

        # Collapse multiple spaces
        date_str = re.sub(r'\s+', ' ', date_str).strip()
        
        # Handle text month formats: "Oct 24 2024", "Oct - 24 - 2024", "Oct-24-2024"
        text_match = re.match(
            r'^([A-Za-z]{3,})\s*[-/\s]*(\d{1,2})(?:\s*[-/\s]*(\d{2,4}))?$', 
            date_str
        )
        if text_match:
            month_name, day, year = text_match.groups()
            month = MONTHS.get(month_name[:3].title())
            
            if not month:
                logger.warning(f"Invalid month name: {month_name}")
                return "null"
            
            day = int(day)
            
            # Validate ranges
            if int(month) < 1 or int(month) > 12:
                return "null"
            if day < 1 or day > 31:
                return "null"
            
            # Return with or without year
            if year:
                year_int = int(year)
                month_int, day_adj, year_adj = adjust_date(int(month), day, year_int)
                return f"{month_int:02d}/{day_adj:02d}/{str(year_adj)[-2:]}"
            else:
                return f"{int(month):02d}/{day:02d}"

        # Replace dots and hyphens with slashes for consistent parsing
        date_str = date_str.replace('.', '/').replace('-', '/')
        parts = date_str.split('/')

        # Validate all parts are numeric before processing
        if not all(part.strip().isdigit() for part in parts if part.strip()):
            logger.warning(f"Date contains non-numeric parts: {date_str}")
            return "null"

        # Two parts: MM/DD (no year)
        if len(parts) == 2:
            month, day = int(parts[0]), int(parts[1])
            
            # Validate ranges
            if month < 1 or month > 12:
                logger.warning(f"Invalid month: {month}")
                return "null"
            if day < 1 or day > 31:
                logger.warning(f"Invalid day: {day}")
                return "null"
            
            # Return as MM/DD format
            return f"{month:02d}/{day:02d}"

        # Three parts: MM/DD/YY or MM/DD/YYYY
        if len(parts) == 3:
            month, day, year = map(int, parts)
            
            # Validate ranges
            if month < 1 or month > 12:
                logger.warning(f"Invalid month: {month}")
                return "null"
            if day < 1 or day > 31:
                logger.warning(f"Invalid day: {day}")
                return "null"
            
            # Adjust for impossible dates
            month, day, year = adjust_date(month, day, year)
            
            # Return with 2-digit year
            return f"{month:02d}/{day:02d}/{str(year)[-2:]}"

        # If format doesn't match, return null
        logger.warning(f"Date format doesn't match: {date_str}")
        return date_str

    except Exception as e:
        logger.error(f"Error processing date '{date_str}': {e}")
        return "null"


def adjust_date(month, day, year=None):
    """
    Adjust ONLY invalid date values (like Feb 31) to nearest valid date.
    Does NOT modify dates based on current date - past/future dates are valid.
    """
    current_year = datetime.now().year
    year_to_use = year if year is not None else current_year
    
    # Clamp month to valid range (1-12)
    month = min(max(month, 1), 12)
    
    # Clamp day to valid range for the given month/year
    day = max(day, 1)
    
    # Calculate last valid day of the month
    from datetime import date, timedelta
    if month == 12:
        last_day = date(year_to_use, 12, 31)
    else:
        last_day = (date(year_to_use, month + 1, 1) - timedelta(days=1))
    
    day = min(day, last_day.day)
    
    return month, day, year_to_use


# Test cases
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_cases = [
        ("02-09-24", "02/09/24"),
        ("02-09-2024", "02/09/24"),
        ("02/09/24", "02/09/24"),
        ("02/09/2024", "02/09/24"),
        ("Oct 24 2024", "10/24/24"),
        ("Oct - 24 2024", "10/24/24"),
        ("Oct - 24 - 2024", "10/24/24"),
        ("02.09.24", "02/09/24"),
        ("02.09.2024", "02/09/24"),
        ("02/09", "02/09"),
        
        # Edge cases
        ("11/13", "11/13"),
        ("3/21", "03/21"),
        ("", "null"),
        ("null", "null"),
    ]
    
    print("\n" + "="*70)
    print("DATE PREPROCESSING TESTS")
    print("="*70 + "\n")
    
    for input_date, expected in test_cases:
        result = preprocess_date(input_date)
        status = "✅ PASS" if result == expected else f"❌ FAIL (expected: {expected})"
        print(f"{status:15} | Input: {str(input_date):20} | Output: {result}")

def validate_customer_number(customer_number, all_numbers=None):
    """
    Validate and extract customer number based on rules
    Args:
        customer_number: Input number to validate
        all_numbers: Optional list of all numbers for length comparison
    Returns:
        Validated customer number or None if invalid
    """
    if isinstance(customer_number, int):
        customer_number = str(customer_number)

    if isinstance(customer_number, str):
        if '/' in customer_number:
            parts = customer_number.split('/')
            if len(parts) == 2 and parts[1].isdigit():
                customer_number = parts[1]
            else:
                return None
        elif '-' in customer_number:
            return None
        elif not customer_number.isdigit():
            return None
    else:
        return None

    if all_numbers:
        str_numbers = [str(num) for num in all_numbers if num is not None]
        max_length = max(len(num) for num in str_numbers)
        return customer_number if len(customer_number) == max_length else None

    return customer_number

def process_customer_numbers(input_data):
    """
    Process single or multiple customer numbers
    Args:
        input_data: Single number or list of numbers
    Returns:
        List of valid customer numbers
    """
    if isinstance(input_data, list):
        valid_numbers = [validate_customer_number(item, input_data) for item in input_data]
        return [num for num in valid_numbers if num is not None]
    else:
        valid_number = validate_customer_number(input_data, [input_data])
        return [valid_number] if valid_number is not None else []

def resize_large_image(img: Image.Image, max_pixels: int = 178956970) -> Image.Image:
    """
    Resize image if it exceeds maximum pixel limit
    Args:
        img: Input PIL Image
        max_pixels: Maximum allowed pixels
    Returns:
        Resized PIL Image if necessary
    """
    width, height = img.size
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

# type conversion helpers
def safe_int_conversion(value, default=0):
    """
    Safely convert OCR output to integer, handling various edge cases.
    
    Args:
        value: Input value from OCR (can be int, float, str, or None)
        default: Default value to return if conversion fails
        
    Returns:
        Integer value or default
        
    Examples:
        safe_int_conversion(40.375) -> 40
        safe_int_conversion("40.375") -> 40
        safe_int_conversion("40") -> 40
        safe_int_conversion("null") -> 0
        safe_int_conversion(None) -> 0
    """
    try:
        # Handle None and empty values
        if value is None or value == "":
            return default
            
        # Handle string values
        if isinstance(value, str):
            value = value.strip().lower()
            # Handle common null/empty representations
            if value in ["null", "empty", "n/a", "na", "none", ""]:
                return default
            # Remove any whitespace and convert
            value = value.replace(" ", "")
        
        # Convert to float first (handles both int and float strings)
        # Then convert to int (truncates decimal part)
        return int(float(value))
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to convert '{value}' to int: {e}. Using default: {default}")
        return default

def safe_float_conversion(value, default=None):
    """
    Safely convert any value to float
    
    Args:
        value: Value to convert (can be str, int, float, or any type)
        default: Default value if conversion fails
    
    Returns:
        float or default value
    """
    if value in ["null", "empty", "N/A", "", None]:
        return default
    
    try:
        # Handle string with commas
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default

def safe_str_conversion(value, default="null"):
    """
    Safely convert OCR output to string, handling various edge cases.
    
    Args:
        value: Input value from OCR (can be any type)
        default: Default value to return if conversion fails
        
    Returns:
        String value or default
    """
    try:
        if value is None:
            return default
        
        # Convert to string
        str_value = str(value).strip().lower()
        
        # Handle empty strings
        if str_value == "":
            return default
            
        return str_value
        
    except Exception as e:
        logger.warning(f"Failed to convert '{value}' to str: {e}. Using default: {default}")
        return default


def safe_numeric_operation(val1, val2, operation="add", default=0):
    """
    Safely perform numeric operations on OCR values.
    
    Args:
        val1: First value
        val2: Second value
        operation: Operation to perform ("add", "subtract", "multiply", "divide")
        default: Default value if operation fails
        
    Returns:
        Result of operation or default value
    """
    try:
        # Convert both values to integers
        num1 = safe_int_conversion(val1, 0)
        num2 = safe_int_conversion(val2, 0)
        
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            return num1 / num2 if num2 != 0 else default
        else:
            logger.warning(f"Unknown operation: {operation}")
            return default
            
    except Exception as e:
        logger.error(f"Error in numeric operation {operation}({val1}, {val2}): {e}")
        return default


def normalize_ocr_data(ocr_data: dict, field_types: dict) -> dict:
    """
    Normalize OCR data based on expected field types
    
    Args:
        ocr_data: Raw OCR response data
        field_types: Dictionary mapping field names to expected types (int, str, float)
    
    Returns:
        Normalized OCR data
    """
    normalized = {}
    
    for field, expected_type in field_types.items():
        value = ocr_data.get(field)
        
        if expected_type == int:
            normalized[field] = safe_int_conversion(value, "null")
        elif expected_type == float:
            normalized[field] = safe_float_conversion(value, "null")
        elif expected_type == str:
            normalized[field] = safe_str_conversion(value, "null")
        else:
            normalized[field] = value
    
    # Copy any remaining fields not in field_types
    for field, value in ocr_data.items():
        if field not in normalized:
            normalized[field] = value
    
    return normalized

def heuristic_stamp_crop_base64(image: str):
    """
    Heuristic fallback to crop stamp region from BOL image
    Uses bottom-right quadrant as stamp is typically located there
    
    Args:
        image: Base64 encoded image string
    
    Returns:
        Base64 encoded cropped stamp region or None if crop fails
    """
    try:
        pil_img = base64topil(image)
        width, height = pil_img.size
        
        # Crop bottom-right quadrant (typical stamp location)
        # Taking 40% from right and 30% from bottom
        left = int(width * 0.6)
        top = int(height * 0.7)
        right = width
        bottom = height
        
        cropped = pil_img.crop((left, top, right, bottom))
        
        # Basic quality check - ensure cropped region is not too small
        if cropped.width >= 50 and cropped.height >= 50:
            return convert_image_to_base64(cropped)
        
        return None
        
    except Exception as e:
        print(f"Error in heuristic stamp crop: {e}")
        return None