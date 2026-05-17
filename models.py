import os
from pydantic import BaseModel, Field, HttpUrl, FilePath, FileUrl, field_validator
import mimetypes
from pathlib import Path
from typing import Optional, Union, Dict, List
from datetime import date
from PIL import Image
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class InterVL2Request(BaseModel):
    image_url_or_path_or_base64:str
    prompt:str

class InterVL2Response(BaseModel):
    response:str

class BillOfLadings(BaseModel):
    bill_no: str = Field("", description="Unique bill of lading number")
    date: Optional[str] = Field(None, description="The date from the bill of lading")

    @field_validator("bill_no", mode="before")
    @classmethod
    def _cast_bill_no_to_str(cls, v):
        # if it's None, default to empty string; otherwise str() it
        if v is None:
            return ""
        return str(v)

class CarrierInfo(BaseModel):
    carrier_name: str = Field("", description="Name of the carrier")

class Stamp(BaseModel):
    pod_date: Optional[str] = Field(
        None, description="Date of the POD, in YYYY-MM-DD format"
    )
    pod_sign: Optional[str] = Field(
        None, description="Whether signature was present or not ('yes', 'no')"
    )
    stamp_exist: Optional[str] = Field(
        None, description="Whether stamp was present or not ('yes', 'no')"
    )
    damage: Optional[Union[int, str]] = Field(
        None, description="Number of items damaged (non-negative integer or '-')"
    )
    over: Optional[Union[int, str]] = Field(
        None, description="Number of items over (non-negative integer or '-')"
    )
    short: Optional[Union[int, str]] = Field(
        None, description="Number of items short (non-negative integer or '-')"
    )
    refused: Optional[Union[int, str]] = Field(
        None, description="Number of items refused (non-negative integer or '-')"
    )
    seal_intact: Optional[str] = Field(
        None, description="Whether the seal was intact ('yes', 'no')"
    )
    damaged_kept: Optional[Union[int, str]] = Field(
        None, description="Number of damaged cartons kept (non-negative integer or '-')"
    )
    roc_damaged: Optional[Union[int, str]] = Field(
        None, description="Record or acknowledgment of damage (non-negative integer or '-')"
    )
    total_received: Optional[Union[int, str]] = Field(
        None, description="Number of cartons received (non-negative integer or '-')"
    )
    notation_exist: Optional[str] = Field(
        None, description="Whether notation was present or not ('yes', 'no')"
    )

class CustomerOrderInfo(BaseModel):
    total_order_quantity: int = Field(0, description="Total number of packages handled")


class Signatures(BaseModel):

    receiver_signature: str = Field(
        default=None,
        pattern="^(yes|no)$",
        description="Indicates whether the receiver has provided a signature. Accepts 'yes' or 'no'."
    )

    @field_validator('receiver_signature', mode='before')
    @classmethod
    def handle_null_string(cls, v):
        """Convert 'null' string and other invalid values to 'no'"""
        if v in ['null', 'N/A', 'empty', '', None]:
            return 'no'
        return v

class DeliveryReceipt(BaseModel):

    
    pod_date: Optional[str] = Field(
        None, description="Date of the POD, in YYYY-MM-DD format"
    )
    pod_sign: Optional[str] = Field(
        None, description="Whether signature was present or not ('yes', 'no')"
    )
    total_received: Optional[Union[List[int] ,int, str]] = Field(
        None, description="Number of cartons received (non-negative integer or 'empty')"
    )
    refused: Optional[Union[List[int], int, str]] = Field(
        None, description="Number of items refused (non-negative integer or 'empty')"
    )
    damage: Optional[Union[List[int], int, str]] = Field(
        None, description="Number of items damaged (non-negative integer or 'empty')"
    )
    customer_order_num: Optional[Union[List[int], List[str], int, str]] = Field(
        None, description="Customer number written on the receipt (non-negative integer or 'empty')"
    )


class OCRRequest(BaseModel):
    id: str = Field(..., alias="_id")
    file_url_or_path: str = Field(
        ..., description="URL or file path to the OCR"
    )

    @field_validator("file_url_or_path")
    def validate_file_url_or_path(cls, value):
        # If it's a URL, check its MIME type
        if isinstance(value, HttpUrl):
            # Allow only common image extensions in the URL
            if not value.path.lower().endswith((".pdf", ".PDF")):
                raise ValueError("URL must point to a valid pdf file.")
        elif isinstance(value, Path):
            # Validate file existence and MIME type for a local file
            if not os.path.exists(value):
                raise ValueError(f"The file path '{value}' does not exist.")
            mime_type, _ = mimetypes.guess_type(value)
            if mime_type is None or not mime_type.startswith("application/pdf"):
                raise ValueError(
                    f"The file path '{value}' must point to a valid pdf file."
                )
        return value
        
    # class Config:
    #     allow_population_by_field_name = True  

class OCRResponse(BaseModel):
    bill_of_lading: BillOfLadings = Field(
        BillOfLadings(), description="Details of the bill of lading"
    )
    # carrier: CarrierInfo = Field(CarrierInfo(), description="Details of the carrier")
    stamp: Stamp = Field(Stamp(), description="Details of the stamp")
    customer_order_info: CustomerOrderInfo = Field(
        CustomerOrderInfo(), description="Details of the customer order"
    )
    signatures: Signatures = Field(
        Signatures(), description="Details of the signatures"
    )
    


class Regions(BaseModel):
    bill_of_lading: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing Bill of Ladings"
    )
    # carrier: Optional[Image.Image] = Field(
    #     None, description="Region containing carrier info"
    # )
    stamp: Optional[Union[Image.Image, str]] = Field(None, description="Region containing stamp")
    customer_order_info: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing order info"
    )
    signatures: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )

    class Config:
        arbitrary_types_allowed = True

# {
    # "pod_date": "10/23/2024",
    # "pod_sign": "yes",
    # "total_received": 205,
    # "damage": "empty",
    # "refused": "empty"
    # }

class OCRBatch(BaseModel):
    image: str
    prompt: str

class OCRBatch(BaseModel):
    page_type: str
    region_name: str
    prompt: str
    image: str
    stamp_exist:Optional[str]=None

class OCRBatchResponse(BaseModel):
    page_type: str
    region_name: str
    ocr_response: dict
    stamp_exist:Optional[str]=None

class ReceiptRegions(BaseModel):
    pod_date: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing date"
    )
    # carrier: Optional[Image.Image] = Field(
    #     None, description="Region containing carrier info"
    # )
    customer_order_num: Optional[Union[Image.Image, str]] = Field(None, description="Region containing stamp")
    total_received: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing order info"
    )
    damage: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )
    refused: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )
    pod_sign: Optional[Union[Image.Image, str]] = Field(
        None, description="Region containing signatures"
    )

    class Config:
        arbitrary_types_allowed = True


class PODResponse(BaseModel):
    B_L_Number: Optional[Union[str, int]] = None
    Stamp_Exists: Optional[str] = None
    Seal_Intact: Optional[str] = None
    POD_Date: Optional[str] = None
    Signature_Exists: Optional[str] = None
    Issued_Qty: Optional[Union[str, int]] = None
    Received_Qty: Optional[Union[str, int]] = None
    Damage_Qty: Optional[Union[str, int]] = None
    Short_Qty: Optional[Union[str, int]] = None
    Over_Qty: Optional[Union[str, int]] = None
    Refused_Qty: Optional[Union[str, int]] = None
    Customer_Order_Num: Optional[Union[str, int, List[Union[str, int]]]] = None  # Accept list too
    id: str = Field(..., alias="_id")


    class Config:
        populate_by_name = True

# Template sync models
class TemplateSyncRequest(BaseModel):
    """Request model for template sync endpoint"""
    action: str = Field(..., description="Action: 'add', 'remove', or 'update'")
    template: Optional[Dict[str, Any]] = Field(None, description="Template data (for add/update)")
    template_id: Optional[str] = Field(None, description="Template ID (for remove)")


class TemplateSyncResponse(BaseModel):
    """Response model for template sync endpoint"""
    success: bool
    message: str
    template_id: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


# Enhanced OCR models
class ClassificationDetails(BaseModel):
    """Classification details from YOLO models"""
    primary_model_prediction: str = Field(..., description="Primary category (BOL/Receipt/Others)")
    primary_confidence: float = Field(..., description="Confidence of primary prediction")
    secondary_model_prediction: Optional[str] = Field(None, description="Secondary category if applicable")
    secondary_confidence: Optional[float] = Field(None, description="Confidence of secondary prediction")


class SuggestedTemplate(BaseModel):
    """Suggested template with match score"""
    template_id: str
    template_name: str
    match_score: float
    priority: int


class EnhancedOCRResponse(BaseModel):
    """
    Enhanced OCR Response with template information
    EXTENDS existing PODResponse model
    """
    # Original fields (keep all existing fields from PODResponse)
    B_L_Number: str = ""
    Stamp_Exists: str = ""
    Seal_Intact: str = "null"
    POD_Date: str = ""
    Signature_Exists: str = ""
    Issued_Qty: int = 0
    Received_Qty: Optional[Union[str, int]] = ""
    Damage_Qty: str = "0"
    Short_Qty: str = "0"
    Over_Qty: str = "0"
    Refused_Qty: str = "null"
    Customer_Order_Num: str = "null"

    # Sticker-related fields (for SK1/SK2/SK3 cases)
    sticker_exist: str = Field("null", description="Whether sticker exists ('yes', 'no', 'null')")
    # receipt_exist: Placeholder for future receipt detection feature (another dev's task)
    # For sticker cases, this remains "null"
    receipt_exist: str = Field("null", description="Whether receipt exists ('yes', 'no', 'null')")
    appointment_id: str = Field("null", description="Appointment ID if available")
    delivery_confirmed_by: str = Field("null", description="Who confirmed delivery")

    # SK3-specific fields (only populated when SK3 sticker case is detected, otherwise null)
    Equip_ID: Optional[str] = Field(None, description="SK3: Equipment/Trailer ID from delivery sticker")
    Equip_Arrival_Date: Optional[str] = Field(None, description="SK3: Equipment arrival date (MM/DD/YY)")
    Carrier: Optional[str] = Field(None, description="SK3: Carrier code or name from delivery sticker")
    
    # NEW FIELDS - Added _id and template_identifier
    id: Optional[str] = Field(None, alias="_id", description="Request _id from backend")
    template_id: Optional[str] = Field(None, description="Template's MongoDB _id (not template_id string)")
    template_identifier: Optional[str] = Field(None, description="Template's template_id string (e.g., STAMP_STANDARD_V555)")
    confidence: float = Field(0.0, description="Template match confidence score")
    processing_time: int = Field(0, description="Processing time in milliseconds")
    classification_details: Optional[ClassificationDetails] = Field(None, description="Classification details")
    suggested_templates: List[SuggestedTemplate] = Field(default_factory=list, description="Suggested templates (only populated when confidence < 0.75)")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")
    class Config:
        populate_by_name = True


# Template test models
class TemplateTestRequest(BaseModel):
    """Request model for template testing endpoint"""
    file_url: str = Field(..., description="URL of document to test")
    template: Dict[str, Any] = Field(..., description="Complete template JSON")
    user_id: Optional[str] = Field(None, description="User ID for logging")


class TemplateTestResponse(BaseModel):
    """Response model for template testing"""
    success: bool
    extracted_data: Optional[EnhancedOCRResponse] = None
    error: Optional[str] = None
    processing_time: int = 0
    regions_metadata: Optional[List[Dict]] = []


# Enhanced OCR Request (extends existing OCRRequest)
class EnhancedOCRRequest(BaseModel):
    """
    Enhanced OCR Request
    Supports normal flow and unregistered document processing
    """
    file_url: str = Field(..., description="URL of document to process")
    user_id: Optional[str] = Field(None, description="User ID for logging")
    template_id: Optional[str] = Field(None, description="Template ID for unregistered doc processing")
    id: Optional[str] = Field(None, alias="_id", description="Request _id from backend")
    
    class Config:
        populate_by_name = True  # ADDED - Allow both 'id' and '_id'
        json_schema_extra = {
            "example": {
                "file_url": "https://example.com/document.pdf",
                "user_id": "user123",
                "template_id": None,
                "_id": "12345"
            }
        }


# Memory store stats
class MemoryStoreStats(BaseModel):
    """Statistics about template memory store"""
    total_templates: int
    categories: List[str]
    templates_by_category: Dict[str, int]