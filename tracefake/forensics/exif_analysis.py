import exifread

def extract_exif(image_file):
    """
    Extracts EXIF metadata from an image file.
    
    Args:
        image_file: File path or file-like object (bytes) with read() method.
        
    Returns:
        dict: A dictionary of key EXIF tags and their values.
    """
    try:
        if hasattr(image_file, 'read'):
            image_file.seek(0)
            tags = exifread.process_file(image_file, details=False)
            image_file.seek(0) # Reset after reading
        else:
            with open(image_file, 'rb') as f:
                tags = exifread.process_file(f, details=False)

        # Filter and format interesting tags
        results = {}
        target_tags = [
            'Image Model', 'Image Make', 'Image Software', 
            'Image DateTime', 'EXIF DateTimeOriginal',
            'EXIF ExposureTime', 'EXIF FNumber', 'EXIF ISOSpeedRatings'
        ]

        for tag_name in tags.keys():
            # If the tag is in our target list (partial match allows catching different formats)
            # or if we want to grab everything that looks relevant:
            if tag_name in target_tags or 'Date' in tag_name or 'Software' in tag_name or 'Model' in tag_name:
                val = tags[tag_name]
                results[tag_name] = str(val)
        
        if not results:
            return {"Info": "No EXIF metadata found."}
            
        return results

    except Exception as e:
        return {"Error": f"Failed to extract EXIF: {str(e)}"}
