# validation.py
import re
from typing import Dict, Any
from datetime import datetime

class StampDataValidator:
    """Validate stamp extraction results"""
    
    @staticmethod
    def validate_stamp_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean stamp data"""
        cleaned = {}
        
        # Validate binary fields
        cleaned['stamp_exist'] = StampDataValidator._validate_yes_no(
            data.get('stamp_exist'), default='no'
        )
        cleaned['notation_exist'] = StampDataValidator._validate_yes_no(
            data.get('notation_exist'), default='no'
        )
        cleaned['seal_intact'] = StampDataValidator._validate_yes_no_empty_null(
            data.get('seal_intact')
        )
        cleaned['pod_sign'] = StampDataValidator._validate_yes_no_empty_null(
            data.get('pod_sign')
        )
        
        # Validate date
        cleaned['pod_date'] = StampDataValidator._validate_date(
            data.get('pod_date')
        )
        
        # Validate numeric fields
        for field in ['total_received', 'damage', 'short', 'over', 'refused',
                     'roc_damaged', 'damaged_kept']:
            cleaned[field] = StampDataValidator._validate_numeric(data.get(field))
        
        return cleaned
    
    @staticmethod
    def _validate_yes_no(value: Any, default: str = 'null') -> str:
        """Validate yes/no fields"""
        if value is None:
            return default
        value_str = str(value).lower().strip()
        if value_str in ['yes', 'y', 'true', '1']:
            return 'yes'
        elif value_str in ['no', 'n', 'false', '0']:
            return 'no'
        return default
    
    @staticmethod
    def _validate_yes_no_empty_null(value: Any) -> str:
        """Validate yes/no/empty/null fields"""
        if value is None or value == 'null':
            return 'null'
        value_str = str(value).lower().strip()
        if value_str in ['yes', 'y', 'true', '1']:
            return 'yes'
        elif value_str in ['no', 'n', 'false', '0']:
            return 'no'
        elif value_str in ['empty', '', 'blank']:
            return 'empty'
        return 'null'
    
    @staticmethod
    def _validate_date(value: Any) -> str:
        """Validate date format"""
        if value is None or str(value).lower() in ['null', 'none']:
            return 'null'
        value_str = str(value).strip()
        if not value_str or value_str.lower() in ['empty', 'blank']:
            return 'empty'
        # Check date patterns
        if re.match(r'\d{1,2}[-/]\d{1,2}[-/]?\d{0,4}', value_str):
            return value_str
        return 'null'
    
    @staticmethod
    def _validate_numeric(value: Any) -> Any:
        """Validate numeric fields"""
        if value is None or str(value).lower() in ['null', 'none']:
            return 'null'
        value_str = str(value).strip()
        if not value_str or value_str.lower() in ['empty', 'blank']:
            return 'empty'
        try:
            numeric_match = re.search(r'-?\d+', value_str)
            if numeric_match:
                num = int(numeric_match.group())
                if 0 <= num <= 10000:
                    return num
        except (ValueError, AttributeError):
            pass
        return 'null'