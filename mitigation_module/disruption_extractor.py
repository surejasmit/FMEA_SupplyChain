"""
Disruption Information Extractor
Uses Claude 3.5 Sonnet for multimodal input processing and JSON extraction
Handles: Text, CSV, Images (OCR), Emails, PDFs
"""

import json
import logging
import re
from typing import Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator
import pandas as pd

logger = logging.getLogger(__name__)

# Import dynamic route lookup for non-hardcoded cities
# This feature allows the extractor to resolve city names into routes
# at runtime. Failure here should not impact OCR functionality.
try:
    from .dynamic_network import get_routes_for_city
    DYNAMIC_ROUTING_AVAILABLE = True
    logger.info("Dynamic routing module loaded successfully.")
except ImportError as e:
    DYNAMIC_ROUTING_AVAILABLE = False
    logger.warning("Dynamic routing not available. Will use mapping config only. ImportError: %s", e)

# OCR support is optional and handled completely independently.
# We only attempt to import easyocr and set OCR_AVAILABLE accordingly.
# A missing dynamic_network module must never affect this value.
try:
    import easyocr
    OCR_AVAILABLE = True
    logger.info("OCR engine (easyocr) is available.")
except ImportError as e:
    OCR_AVAILABLE = False
    logger.warning(
        "OCR engine not available. Install via 'pip install easyocr' to enable image extraction. ImportError: %s",
        e,
    )


class DisruptionEvent(BaseModel):
    """
    Validated disruption event model
    Ensures clean output regardless of messy input
    """
    target_route_id: int = Field(..., ge=1, description="Route ID affected")
    impact_type: str = Field(..., description="Type of disruption (flood, strike, accident, etc.)")
    cost_multiplier: float = Field(..., ge=1.0, description="Cost multiplication factor")
    severity_score: int = Field(..., ge=1, le=10, description="Severity rating (1-10)")
    
    @validator('impact_type')
    def normalize_impact_type(cls, v):
        """Normalize impact type to lowercase"""
        return v.lower().strip()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'target_route_id': self.target_route_id,
            'impact_type': self.impact_type,
            'cost_multiplier': self.cost_multiplier,
            'severity_score': self.severity_score
        }


class DisruptionExtractor:
    """
    Multimodal disruption information extractor
    Uses Claude 3.5 Sonnet (via API) or rule-based fallback
    """
    
    def __init__(self, config_path: str = "mitigation_module/mapping_config.json"):
        """
        Initialize extractor with location mapping
        
        Args:
            config_path: Path to mapping configuration JSON
        """
        self.config_path = Path(config_path)
        self.mapping_config = self._load_mapping_config()
        self.ocr_reader = None
        
        if OCR_AVAILABLE:
            # create the OCR reader lazily; log success
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("Initialized OCR reader for image extraction.")
    
    def _load_mapping_config(self) -> Dict:
        """Load location to Route ID mapping"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config not found: {self.config_path}. Using defaults.")
            return {"mappings": {"locations": {}}, "impact_types": {}}
    
    def extract_from_text(self, text: str) -> List[DisruptionEvent]:
        """
        Extract disruption from plain text
        
        Args:
            text: Raw text input
            
        Returns:
            List of validated disruption events
        """
        logger.info(f"Extracting from text: {text[:100]}...")
        
        # Rule-based extraction (fallback)
        disruptions = self._rule_based_extraction(text)
        
        # TODO: Add Claude 3.5 Sonnet API call when API key is provided
        # disruptions = self._llm_extraction(text)
        
        return [DisruptionEvent(**d) for d in disruptions]
    
    def extract_from_csv(self, file_path: str) -> List[DisruptionEvent]:
        """
        Extract disruptions from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of validated disruption events
        """
        df = pd.read_csv(file_path)
        disruptions = []
        
        # Check if CSV has required columns
        required_cols = ['target_route_id', 'impact_type', 'cost_multiplier', 'severity_score']
        if all(col in df.columns for col in required_cols):
            # Direct mapping from CSV
            for idx, row in df.iterrows():
                try:
                    event = DisruptionEvent(
                        target_route_id=int(row['target_route_id']),
                        impact_type=str(row['impact_type']),
                        cost_multiplier=float(row['cost_multiplier']),
                        severity_score=int(row['severity_score'])
                    )
                    disruptions.append(event)
                except Exception as e:
                    error_msg = f"Failed to parse CSV row {idx}: {row.to_dict()}. Error: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e
        else:
            # Extract from text columns
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            for _, row in df.iterrows():
                text = ' '.join(str(row[col]) for col in text_cols)
                disruptions.extend(self.extract_from_text(text))
        
        return disruptions
    
    def extract_from_image(self, image_path: str) -> List[DisruptionEvent]:
        """
        Extract disruptions from image using OCR
        
        Args:
            image_path: Path to image file (PNG, JPG)
            
        Returns:
            List of validated disruption events
        """
        if not OCR_AVAILABLE or not self.ocr_reader:
            raise RuntimeError("OCR not available. Install: pip install easyocr")
        
        # Extract text using OCR
        results = self.ocr_reader.readtext(image_path)
        text = '\n'.join([result[1] for result in results])
        
        logger.info(f"OCR extracted text: {text[:200]}...")
        
        # Process extracted text
        return self.extract_from_text(text)
    
    def _rule_based_extraction(self, text: str) -> List[Dict]:
        """
        TRULY DYNAMIC extraction - Extracts actual route numbers from user text
        NO PREDEFINED SCENARIOS - Responds to what user actually writes
        """
        # `re` is imported at module level for reuse.
        text_lower = text.lower()
        disruptions = []
        
        # log the incoming text fragment for troubleshooting
        logger.debug("[EXTRACTOR] Processing input: %s", text[:100])
        
        # STEP 1: Extract explicitly mentioned route numbers from text
        # We only consider routes when they are introduced with a keyword
        # such as "route", "routes" or the abbreviation "r". This avoids
        # the previous catch-all \b(\d+)\b fallback which pulled in unrelated
        # numbers (years, costs, phone numbers, etc.).
        #
        # Supported patterns:
        #   "route 12"  => 12
        #   "R45"       => 45
        #   "routes 10, 20 and 30" => [10,20,30]
        #
        # We intentionally *do not* fall back to grabbing every standalone
        # digit sequence; if no context is provided we prefer an empty result
        # and rely on mapping/config or dynamic lookup instead.
        route_pattern = re.compile(
            r"\b(?:route|r)\s*(\d+)\b"              # single route reference
            r"|\b(?:routes?)\s*((?:\d+(?:\s*(?:,|and)\s*)?)+)\b",  # list of routes
            re.IGNORECASE,
        )
        matches = route_pattern.finditer(text_lower)
        
        affected_routes = []
        for match in matches:
            if match.group(1):  # Single route: "route 3" or "r3"
                affected_routes.append(int(match.group(1)))
            elif match.group(2):  # Multiple routes: "routes 2, 5, and 8"
                # Extract all numbers from the matched portion
                numbers = re.findall(r"\d+", match.group(2))
                affected_routes.extend([int(n) for n in numbers])
        
        # STEP 2: If no route numbers found, try location-based extraction from mapping config
        if not affected_routes:
            # Use the loaded mapping config (supports many more locations)
            mappings = self.mapping_config.get('mappings', {}).get('locations', {})
            
            # Try all mappings (case-insensitive)
            for location, routes in mappings.items():
                if location.lower() in text_lower:
                    affected_routes.extend(routes)
                    logger.debug("[EXTRACTOR] Mapped location '%s' to routes %s", location, routes)
                    break
            
            # STEP 2b: If still no match and dynamic routing is available, try dynamic lookup
            if not affected_routes and DYNAMIC_ROUTING_AVAILABLE:
                # Extract potential city names (capitalized words that might be cities)
                potential_cities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text)
                
                for city in potential_cities:
                    try:
                        # Attempt dynamic route resolution
                        dynamic_routes = get_routes_for_city(city, include_multihop=False)
                        if dynamic_routes:
                            affected_routes.extend(dynamic_routes[:2])  # Use first 2 routes
                            logger.debug("[EXTRACTOR] Dynamically resolved '%s' to routes %s", city, dynamic_routes[:2])
                            break
                    except Exception:
                        # ignore and continue with other candidate cities
                        continue
        
        # STEP 3: No generic number extraction
        # We removed the previous catch-all that treated every digit sequence as a
        # potential route ID. That behaviour produced false positives from years,
        # costs, or other unrelated numeric data. If the text does not explicitly
        # mention a route or map to a location, we do *not* guess at standalone
        # numbers. This keeps the output clean and predictable.
        #
        # (If future requirements demand a restricted fallback, it should still
        # require a contextual keyword such as "route" preceding the number.)
        #
        # At this point affected_routes remains empty and later code will either
        # map a location or return an empty list with a warning.
        # no debug print needed here
        
        # STEP 4: Determine severity/multiplier from keywords (DYNAMIC based on text)
        cost_multiplier = 1.5  # Base default
        severity_score = 5
        impact_type = "Disruption"
        
        # High severity keywords
        if any(word in text_lower for word in ['collapse', 'catastrophic', 'critical', 'severe', 'major', 'closed', 'shutdown']):
            cost_multiplier = 15.0
            severity_score = 10
            impact_type = "Critical"
        elif any(word in text_lower for word in ['fire', 'explosion', 'toxic', 'chemical', 'spill']):
            cost_multiplier = 10.0
            severity_score = 9
            impact_type = "Hazardous"
        elif any(word in text_lower for word in ['strike', 'protest', 'blockade']):
            cost_multiplier = 6.0
            severity_score = 7
            impact_type = "Labor/Civil"
        elif any(word in text_lower for word in ['accident', 'crash', 'collision']):
            cost_multiplier = 4.0
            severity_score = 6
            impact_type = "Accident"
        elif any(word in text_lower for word in ['delay', 'slow', 'congestion', 'traffic']):
            cost_multiplier = 2.0
            severity_score = 4
            impact_type = "Delay"
        
        # Extract explicit multiplier if mentioned (e.g., "10x cost" or "cost increased by 5")
        multiplier_pattern = r'(\d+(?:\.\d+)?)\s*(?:x|times|multiplier)'
        multiplier_match = re.search(multiplier_pattern, text_lower)
        if multiplier_match:
            cost_multiplier = float(multiplier_match.group(1))
            logger.debug("[EXTRACTOR] Found explicit multiplier in text: %sx", cost_multiplier)
        
        # GRACEFUL FALLBACK if no routes could be extracted
        if not affected_routes:
            warning_msg = (
                f"WARNING: Could not extract route information from: '{text[:100]}...'\n"
                f"No explicit route numbers found and location not recognized.\n"
                f"Returning empty disruption list. To fix:\n"
                f"  1. Specify route numbers explicitly (e.g., 'Route 3', 'routes 5 and 7'), OR\n"
                f"  2. Add location to mapping_config.json, OR\n"
                f"  3. Mention a recognized location (check mapping_config.json for available locations)"
            )
            logger.warning("[EXTRACTOR] %s", warning_msg)
            # Return empty list instead of raising error
            return []
        
        logger.debug("[EXTRACTOR] ✓ Extracted Routes: %s", affected_routes)
        logger.debug("[EXTRACTOR] ✓ Impact Type: %s", impact_type)
        logger.debug("[EXTRACTOR] ✓ Cost Multiplier: %sx", cost_multiplier)
        logger.debug("[EXTRACTOR] ✓ Severity: %s/10", severity_score)
        
        # Create disruption for each affected route
        for route_id in set(affected_routes):  # Remove duplicates
            disruptions.append({
                'target_route_id': route_id,
                'impact_type': impact_type,
                'cost_multiplier': cost_multiplier,
                'severity_score': severity_score
            })
        
        return disruptions
        
    def _old_mapping_based_extraction(self, text: str) -> List[Dict]:
        """
        OLD LOGIC - Kept for reference but not used
        """
        text_lower = text.lower()
        disruptions = []
        
        # Map locations to route IDs
        affected_routes = []
        for location, route_ids in self.mapping_config['mappings']['locations'].items():
            if location.lower() in text_lower:
                affected_routes.extend(route_ids)
        
        # Determine impact type
        impact_type = 'accident'  # default
        for imp_type in self.mapping_config['impact_types'].keys():
            if imp_type in text_lower:
                impact_type = imp_type
                break
        
        # Get default multiplier and severity
        impact_config = self.mapping_config['impact_types'].get(
            impact_type,
            {'default_multiplier': 1.5, 'severity_range': [5, 7]}
        )
        
        cost_multiplier = impact_config['default_multiplier']
        severity_score = impact_config['severity_range'][0]
        
        # Adjust based on keywords
        if any(word in text_lower for word in ['severe', 'major', 'critical', 'catastrophic']):
            cost_multiplier = min(cost_multiplier * 1.5, 10.0)
            severity_score = min(severity_score + 2, 10)
        elif any(word in text_lower for word in ['minor', 'slight', 'small']):
            cost_multiplier = max(cost_multiplier * 0.8, 1.0)
            severity_score = max(severity_score - 2, 1)
        
        # Create disruptions for affected routes
        if not affected_routes:
            # NO FALLBACK - Raise error so user sees what's wrong
            error_msg = (
                f"Cannot extract route information from text: '{text[:100]}...'. "
                f"No location keywords found in mapping_config.json. "
                f"Available locations: {list(self.mapping_config['mappings']['locations'].keys())}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        for route_id in set(affected_routes):  # Remove duplicates
            disruptions.append({
                'target_route_id': route_id,
                'impact_type': impact_type,
                'cost_multiplier': cost_multiplier,
                'severity_score': severity_score
            })
        
        logger.info(f"Extracted {len(disruptions)} disruptions: {disruptions}")
        return disruptions
    
    def extract_from_news(self, news_df: pd.DataFrame, 
                         categories: List[str] = ['BUSINESS', 'WORLD NEWS']) -> List[DisruptionEvent]:
        """
        Extract disruptions from historical news dataset
        
        Args:
            news_df: DataFrame with 'category', 'headline', 'short_description'
            categories: News categories to filter
            
        Returns:
            List of validated disruption events
        """
        # Filter relevant categories
        filtered = news_df[news_df['category'].isin(categories)]
        
        disruptions = []
        for _, row in filtered.head(50).iterrows():  # Limit to 50 articles
            text = f"{row.get('headline', '')} {row.get('short_description', '')}"
            
            # Check for transportation/logistics keywords
            if any(keyword in text.lower() for keyword in [
                'transport', 'logistics', 'supply', 'shipping', 'port',
                'highway', 'road', 'traffic', 'strike', 'delay'
            ]):
                disruptions.extend(self.extract_from_text(text))
        
        return disruptions
    
    def validate_and_aggregate(self, disruptions: List[DisruptionEvent]) -> List[Dict]:
        """
        Validate and aggregate disruptions by route
        If multiple disruptions affect same route, use worst case
        
        Args:
            disruptions: List of disruption events
            
        Returns:
            Aggregated disruptions (one per route)
        """
        route_disruptions = {}
        
        for disruption in disruptions:
            route_id = disruption.target_route_id
            
            if route_id not in route_disruptions:
                route_disruptions[route_id] = disruption
            else:
                # Keep disruption with higher cost multiplier
                existing = route_disruptions[route_id]
                if disruption.cost_multiplier > existing.cost_multiplier:
                    route_disruptions[route_id] = disruption
        
        return [d.to_dict() for d in route_disruptions.values()]
