# d:\last-semester\gradution-project\multi-model-rag-app\src\controllers\DataController.py

from controllers.BaseController import BaseController
from fastapi import UploadFile
# Import the actual ResponseSignal enum from its location
from models.enums.ResponseEnums import ResponseSignal
from .ProjectController import ProjectController
import re
import os
import logging # Added for potential logging

logger = logging.getLogger("uvicorn.error") # Setup logger

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        # Define size scale factor (1 MiB = 1024 * 1024 bytes)
        self.size_scale = 1024 * 1024 # 1048576

    def validate_uploaded_file(self, file: UploadFile):
        """
        Validates the uploaded file based on content type and size.

        NOTE: This method uses ResponseSignal members that might not exist
              in the provided ResponseEnums.py (FILE_SIZE_EXCEEDED, FILE_VALIDATED_SUCCESS).
              It also needs to correctly parse FILE_ALLOWED_TYPES from settings.
        """

        # --- FIX NEEDED for Content Type Check ---
        try:
            # Assuming FILE_ALLOWED_TYPES is a comma-separated string like "application/pdf,image/jpeg"
            allowed_types_str = self.app_settings.FILE_ALLOWED_TYPES
            if not allowed_types_str:
                logger.error("Configuration error: FILE_ALLOWED_TYPES is not set in settings.")
                # Use an appropriate existing signal or add a new one for config errors
                return False, ResponseSignal.PROCESSING_FAILED.value # Example fallback

            # Split, strip whitespace, and convert to lowercase for comparison
            allowed_types = [t.strip().lower() for t in allowed_types_str.split(',')]

            # Get file content type and convert to lowercase
            file_content_type = file.content_type.lower() if file.content_type else ""

            if file_content_type not in allowed_types:
                logger.warning(f"Validation failed: Unsupported content type '{file.content_type}'. Allowed: {allowed_types}")
                # Use the defined enum member
                return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
        except Exception as e:
             logger.error(f"Error during file type validation: {e}", exc_info=True)
             return False, ResponseSignal.PROCESSING_FAILED.value # Generic failure
        # --- END FIX NEEDED ---


        # Check File Size
        try:
            # Compare file size in bytes directly with max size in bytes
            # Assumes FILE_MAX_SIZE in .env is in MB
            max_size_bytes = self.app_settings.FILE_MAX_SIZE * self.size_scale
            if file.size > max_size_bytes:
                logger.warning(f"Validation failed: File size {file.size} exceeds max {max_size_bytes} bytes.")
                # --- POTENTIAL FIX NEEDED: Use FILE_SIZE_TOO_LARGE ---
                # return False, ResponseSignal.FILE_SIZE_EXCEEDED.value # This likely causes AttributeError
                return False, ResponseSignal.FILE_SIZE_TOO_LARGE.value # Use the one defined in enum
                # --- END POTENTIAL FIX ---
        except Exception as e:
            logger.error(f"Error during file size validation: {e}", exc_info=True)
            return False, ResponseSignal.PROCESSING_FAILED.value # Generic failure

        # If validation passes
        logger.debug(f"File validation successful for '{file.filename}'.")
        # --- POTENTIAL FIX NEEDED: Use FILE_UPLOAD_SUCCESS ---
        # return True, ResponseSignal.FILE_VALIDATED_SUCCESS.value # This likely causes AttributeError
        return True, ResponseSignal.FILE_UPLOAD_SUCCESS.value # Use an existing success signal
        # --- END POTENTIAL FIX ---

    def generate_unique_filepath(self, orig_file_name: str, project_id: str):
        """
        Generates a unique filename and the full path for storing the uploaded file.
        """
        try:
            random_key = self.generate_random_string() # From BaseController
            # Get project path (ensures directory exists)
            project_path = ProjectController().get_project_path(project_id = project_id)
            cleaned_file_name = self.get_clean_file_name(
                orig_file_name = orig_file_name
                )

            # Construct the unique filename first
            unique_filename = f"{random_key}_{cleaned_file_name}"
            new_file_path = os.path.join(project_path, unique_filename)

            # Check for existence and regenerate if necessary (unlikely with UUID/random string but safe)
            attempts = 0
            max_attempts = 5 # Prevent infinite loop in edge cases
            while os.path.exists(new_file_path) and attempts < max_attempts:
                logger.warning(f"Collision detected for generated path: {new_file_path}. Regenerating...")
                random_key = self.generate_random_string()
                unique_filename = f"{random_key}_{cleaned_file_name}"
                new_file_path = os.path.join(project_path, unique_filename)
                attempts += 1

            if attempts >= max_attempts:
                 raise RuntimeError(f"Failed to generate a unique filepath for {orig_file_name} after {max_attempts} attempts.")

            logger.debug(f"Generated unique path: '{new_file_path}'")
            # Return the full path and the unique filename part
            return new_file_path, unique_filename
        except Exception as e:
            logger.error(f"Error generating unique filepath: {e}", exc_info=True)
            raise # Re-raise the exception to be caught by the route handler


    def get_clean_file_name(self, orig_file_name: str):
        """
        Cleans the original filename to be safe for filesystem usage.
        Removes special characters except underscore, hyphen, and period.
        Replaces spaces with underscores.
        """
        # Remove leading/trailing whitespace
        name = orig_file_name.strip()

        # Replace spaces with underscores first
        name = name.replace(" ", "_")

        # Keep alphanumeric, underscore, hyphen, period. Replace others with underscore.
        # Note: This version keeps '.', allowing file extensions.
        cleaned_file_name = re.sub(r'[^\w.-]', '_', name)

        # Optional: Collapse multiple underscores/hyphens/periods if needed
        # cleaned_file_name = re.sub(r'[_.-]+', '_', cleaned_file_name)

        # Optional: Prevent starting/ending with problematic characters
        # cleaned_file_name = cleaned_file_name.strip('._-')

        # Handle potential empty filename after cleaning
        if not cleaned_file_name or cleaned_file_name == '.':
             return "default_filename" # Provide a default if cleaning removes everything

        return cleaned_file_name

