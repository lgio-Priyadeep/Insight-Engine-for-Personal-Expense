"""
config.py — Finance Engine Configuration
=========================================
Edit CATEGORY_PRIORITY to control which category wins when a remark
matches multiple categories. First entry = highest priority.

Edit CATEGORY_KEYWORDS to extend or refine keyword matching.
All keywords are matched against LOWERCASED, CLEANED remarks.
Multi-word keywords (e.g. "cash withdrawal") are fully supported.
"""
import os

LOG_LEVEL = os.getenv("INSIGHT_LOG_LEVEL", "INFO").upper()
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

if LOG_LEVEL not in VALID_LOG_LEVELS:
    raise ValueError(f"Invalid LOG_LEVEL: {LOG_LEVEL}")

ENABLE_CRASH_DUMPS = os.getenv("INSIGHT_ENABLE_CRASH_DUMPS", "False").lower() == "true"
CRASH_DUMP_DIR = os.getenv("INSIGHT_CRASH_DUMP_DIR", "output/crashes")
ENABLE_PII_DEBUG_LOGS = os.getenv("INSIGHT_ENABLE_PII_DEBUG_LOGS", "False").lower() == "true"

# ---------------------------------------------------------------------------
# Merchant normalisation
# ---------------------------------------------------------------------------
GENERIC_ROUTER_ALIASES: dict[str, str] = {
    # ── UPI / Payments ───────────────────────────────────────────────────────
    r"upi[-/]?\d*":                         "UPI Transfer",
    r"paytm":                               "Paytm",
    r"phonepe":                             "PhonePe",
    r"gpay|google\s?pay":                   "Google Pay",
    r"bhim":                                "BHIM",
    r"amazon\s?pay":                        "Amazon Pay",
    r"cred\b":                              "CRED",
    r"mobikwik":                            "MobiKwik",
    r"payzapp|pay\s?zapp":                  "PayZapp",       # HDFC wallet
    r"freecharge":                          "FreeCharge",
    r"razorpay":                            "Razorpay",
    r"payu\b":                              "PayU",
    r"cashfree":                            "Cashfree",
    r"neft|rtgs|imps":                      "Bank Transfer",
}

SPECIFIC_MERCHANT_ALIASES: dict[str, str] = {
    # ── Food Delivery & QSR ──────────────────────────────────────────────────
    r"swiggy":                              "Swiggy",
    r"zomato":                              "Zomato",
    r"dominos?|domino'?s":                  "Domino's",
    r"kfc":                                 "KFC",
    r"mcdonald'?s?|mcd\b":                 "McDonald's",
    r"subway":                              "Subway",
    r"pizza\s?hut":                         "Pizza Hut",
    r"burger\s?king|bk\b":                  "Burger King",
    r"dunkin'?\s?donuts?|dunkin\b":         "Dunkin' Donuts",
    r"starbucks":                           "Starbucks",
    r"cafe\s?coffee\s?day|ccd\b":           "Café Coffee Day",
    r"chaayos":                             "Chaayos",
    r"haldiram'?s?":                        "Haldiram's",
    r"barbeque\s?nation|bbq\s?nation":      "Barbeque Nation",
    r"wow\s?momo":                          "Wow! Momo",
    r"naturals?\s?ice\s?cream":            "Natural's Ice Cream",

    # ── Grocery & Quick Commerce ─────────────────────────────────────────────
    r"bigbasket|big\s?basket":              "BigBasket",
    r"blinkit|grofers":                     "Blinkit",
    r"zepto\b":                             "Zepto",
    r"dunzo":                               "Dunzo",
    r"jiomart|jio\s?mart":                  "JioMart",
    r"dmart|d-mart|avenue\s?super":         "DMart",
    r"reliance\s?(fresh|smart|retail)":     "Reliance Retail",
    r"big\s?bazaar|bigbazaar":              "Big Bazaar",
    r"nature'?s?\s?basket":                "Nature's Basket",
    r"more\s?(retail|supermarket)":         "More Retail",
    r"spencer'?s?":                         "Spencer's",

    # ── E-commerce ───────────────────────────────────────────────────────────
    r"amazon":                              "Amazon",
    r"flipkart":                            "Flipkart",
    r"myntra":                              "Myntra",
    r"nykaa":                               "Nykaa",
    r"meesho":                              "Meesho",
    r"snapdeal":                            "Snapdeal",
    r"ajio\b":                              "AJIO",
    r"tata\s?cliq|tatacliq":               "Tata CLiQ",
    r"indiamart":                           "IndiaMart",
    r"lenskart":                            "Lenskart",
    r"pepperfry":                           "Pepperfry",
    r"urban\s?ladder":                      "Urban Ladder",
    r"bewakoof":                            "Bewakoof",
    r"firstcry":                            "FirstCry",
    r"boat\s?(lifestyle)?":                "boAt",

    # ── Electronics & Retail ─────────────────────────────────────────────────
    r"croma\b":                             "Croma",
    r"reliance\s?digital":                  "Reliance Digital",
    r"vijay\s?sales":                       "Vijay Sales",
    r"decathlon":                           "Decathlon",
    r"ikea\b":                              "IKEA",
    r"apple\s?(store|india)?":             "Apple Store",

    # ── Fashion & Apparel ────────────────────────────────────────────────────
    r"h\s?&\s?m|h and m":                  "H&M",
    r"zara\b":                              "Zara",
    r"westside":                            "Westside",
    r"lifestyle\s?(store)?":               "Lifestyle",
    r"max\s?fashion":                       "Max Fashion",
    r"pantaloons?":                         "Pantaloons",
    r"v-?mart":                             "V-Mart",
    r"fabindia|fab\s?india":               "FabIndia",
    r"w\s?for\s?woman|aurelia":            "W / Aurelia",

    # ── Transport ────────────────────────────────────────────────────────────
    r"uber":                                "Uber",
    r"ola\b":                               "Ola",
    r"rapido":                              "Rapido",
    r"irctc":                               "IRCTC",
    r"indigo|interglobe":                   "IndiGo",
    r"spicejet|spice\s?jet":               "SpiceJet",
    r"air\s?india|airindia":               "Air India",
    r"vistara":                             "Vistara",
    r"akasa\s?air":                         "Akasa Air",
    r"makemytrip|mmt\b":                    "MakeMyTrip",
    r"goibibo":                             "Goibibo",
    r"yatra\b":                             "Yatra",
    r"redbus|red\s?bus":                    "RedBus",
    r"abhibus":                             "AbhiBus",
    r"ola\s?electric|ola\s?ev":            "Ola Electric",

    # ── Hotels & Stays ───────────────────────────────────────────────────────
    r"oyo\b|oyo\s?rooms":                   "OYO",
    r"treebo":                              "Treebo",
    r"fab\s?hotel|fabhotel":               "FabHotels",
    r"airbnb":                              "Airbnb",
    r"taj\s?(hotel|group)?":               "Taj Hotels",
    r"marriott":                            "Marriott",

    # ── Fuel ─────────────────────────────────────────────────────────────────
    r"iocl|indian\s?oil|indianoil":        "Indian Oil",
    r"bpcl|bharat\s?petroleum":            "Bharat Petroleum",
    r"hpcl|hindustan\s?petroleum":         "Hindustan Petroleum",
    r"reliance\s?(petro|petroleum|fuel)":  "Reliance Fuel",

    # ── Streaming & Entertainment ────────────────────────────────────────────
    r"netflix":                             "Netflix",
    r"amazon\s?prime|prime\s?video":        "Amazon Prime",
    r"hotstar|disneyplus|disney\s?\+":     "Disney+ Hotstar",
    r"spotify":                             "Spotify",
    r"jiocin(ema)?":                        "JioCinema",
    r"sonyliv|sony\s?liv":                 "SonyLIV",
    r"zee\s?5|zee5":                        "ZEE5",
    r"voot\b":                              "Voot",
    r"altbalaji|alt\s?balaji":             "ALTBalaji",
    r"mxplayer|mx\s?player":               "MX Player",
    r"youtube\s?premium|yt\s?premium":     "YouTube Premium",
    r"jio\s?saavn|jiosaavn":               "JioSaavn",
    r"gaana\b":                             "Gaana",
    r"hungama":                             "Hungama",
    r"sun\s?nxt":                           "Sun NXT",

    # ── Telecom ──────────────────────────────────────────────────────────────
    r"airtel":                              "Airtel",
    r"jio\b|jio\s?(prepaid|fiber|mobile)":  "Jio",
    r"bsnl":                                "BSNL",
    r"vodafone|vi\b|voda\b":              "Vi (Vodafone Idea)",
    r"mtnl":                                "MTNL",

    # ── Utilities & DTH ──────────────────────────────────────────────────────
    r"bescom":                              "BESCOM",
    r"tata\s?power":                        "Tata Power",
    r"msedcl|mahavitaran|mseb\b":          "MSEDCL",
    r"kseb\b":                              "KSEB",
    r"cesc\b":                              "CESC",
    r"adani\s?(elec|electricity|power)":   "Adani Electricity",
    r"torrent\s?power":                     "Torrent Power",
    r"tata\s?(sky|play)|tatasky":          "Tata Play",
    r"dish\s?tv|dishtv":                   "Dish TV",
    r"d2h\b|videocon\s?d2h":              "D2H",
    r"sun\s?direct":                        "Sun Direct",
    r"ind(raprastha)?\s?gas|igl\b":        "Indraprastha Gas",
    r"mahanagar\s?gas|mgl\b":             "Mahanagar Gas",
    r"bwssb":                               "BWSSB",

    # ── Healthcare & Pharmacy ────────────────────────────────────────────────
    r"apollo\s?(pharmacy|med|health)?":    "Apollo Pharmacy",
    r"netmeds?":                            "Netmeds",
    r"1mg\b|tata\s?1mg":                   "Tata 1mg",
    r"pharmeasy|pharmEasy":                "PharmEasy",
    r"practo":                              "Practo",
    r"medlife":                             "Medlife",
    r"manipal\s?(hospital|health)":        "Manipal Health",
    r"fortis\s?(hospital|health)?":        "Fortis",
    r"max\s?(hospital|healthcare)":        "Max Healthcare",

    # ── Finance & Insurance ──────────────────────────────────────────────────
    r"lic\b|life\s?insurance\s?corp":      "LIC",
    r"hdfc\s?life":                         "HDFC Life",
    r"sbi\s?life":                          "SBI Life",
    r"icici\s?(pru|prudential|lombard)":   "ICICI Insurance",
    r"bajaj\s?(allianz|finance|finserv)":  "Bajaj Finserv",
    r"policybazaar|policy\s?bazaar":        "PolicyBazaar",
    r"zerodha|kite\b":                      "Zerodha",
    r"groww\b":                             "Groww",
    r"upstox":                              "Upstox",
    r"angel\s?(one|broking)":              "Angel One",
    r"5paisa":                              "5paisa",
    r"smallcase":                           "Smallcase",
    r"cleartax|clear\s?tax":              "ClearTax",
    r"et\s?money|etmoney":                 "ET Money",
    r"kuvera":                              "Kuvera",

    # ── Education & Ed-Tech ──────────────────────────────────────────────────
    r"byju'?s?|byjus":                     "BYJU'S",
    r"unacademy":                           "Unacademy",
    r"vedantu":                             "Vedantu",
    r"upgrad|up\s?grad":                   "upGrad",
    r"coursera":                            "Coursera",
    r"udemy":                               "Udemy",
    r"whitehat\s?jr?|white\s?hat":         "WhiteHat Jr",
    r"great\s?learning":                    "Great Learning",
    r"simplilearn":                         "Simplilearn",

    # ── Gaming & Fantasy Sports ──────────────────────────────────────────────
    r"dream\s?11|dream11":                 "Dream11",
    r"mpl\b|mobile\s?premier\s?league":   "MPL",
    r"winzo":                               "Winzo",
    r"my11circle":                          "My11Circle",
    r"games?\s?24x7":                      "Games24x7",

    # ── Real Estate ──────────────────────────────────────────────────────────
    r"nobroker":                            "NoBroker",
    r"magicbricks":                         "MagicBricks",
    r"99acres|99\s?acres":                 "99acres",
    r"housing\s?\.?\s?com":               "Housing.com",
    r"commonfloor":                         "CommonFloor",
}


# ── Debit Category Priority (highest → lowest) ────────────────────────────────
# Hierarchical semantic lists. Higher lists override lower lists unconditionally.

HIGH_PRIORITY = [
    "finance",      # EMI, loans, insurance — highest intent signal
    "health",       # pharmacy, hospital
    "utilities",    # electricity, internet, gas, dth
]

MEDIUM_PRIORITY = [
    "food",         # zomato, swiggy, restaurants
    "transport",    # uber, ola, fuel
    "shopping",     # amazon, flipkart, retail
    "entertainment",# streaming, gaming
]

LOW_PRIORITY = [
    "atm",          # cash withdrawal
    "transfer",     # self-transfers, NEFT, UPI
]

CATEGORY_PRIORITY = HIGH_PRIORITY + MEDIUM_PRIORITY + LOW_PRIORITY

TIER_MAPPING = {}
for i, cat in enumerate(HIGH_PRIORITY):
    TIER_MAPPING[cat] = {"tier_name": "high", "priority": 100 + i, "confidence": 1.0}
for i, cat in enumerate(MEDIUM_PRIORITY):
    TIER_MAPPING[cat] = {"tier_name": "medium", "priority": 200 + i, "confidence": 0.8}
for i, cat in enumerate(LOW_PRIORITY):
    TIER_MAPPING[cat] = {"tier_name": "low", "priority": 300 + i, "confidence": 0.6}


# ── Debit Category Keywords ───────────────────────────────────────────────────
# Keys must match entries in CATEGORY_PRIORITY exactly.
# Will now heavily leverage lowercase normalised merchant strings.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "food": [
        "swiggy", "zomato", "domino", "kfc", "mcdonald", "subway", "pizza hut",
        "burger king", "dunkin", "starbucks", "coffee", "haldiram", "barbeque", 
        "momo", "ice cream", "bigbasket", "blinkit", "zepto", "dunzo", "jiomart",
        "dmart", "reliance retail", "big bazaar", "grocery", "supermarket",
        "spencer", "spencers",
    ],
    "transport": [
        "uber", "ola", "rapido", "irctc", "indigo", "spicejet", "air india",
        "vistara", "akasa", "makemytrip", "goibibo", "yatra", "redbus", "abhibus",
        "indian oil", "bharat petroleum", "hindustan petroleum", "fuel", "petrol", 
        "diesel", "fastag", "toll",
    ],
    "shopping": [
        "amazon", "flipkart", "myntra", "nykaa", "meesho", "snapdeal", "ajio",
        "tata cliq", "croma", "decathlon", "ikea", "h&m", "zara", "lifestyle",
        "pantaloons", "fabindia", "lenskart", "pepperfry", "apple store", "retail",
    ],
    "utilities": [
        "bescom", "tata power", "msedcl", "kseb", "cesc", "adani", "torrent",
        "electricity", "water", "internet", "broadband", "jio", "airtel", "vi", 
        "vodafone", "bsnl", "gas", "tata play", "dish tv", "d2h", "sun direct", "utility",
    ],
    "health": [
        "apollo", "netmeds", "1mg", "pharmeasy", "practo", "medlife", "hospital",
        "clinic", "pharmacy", "doctor", "diagnostic", "fortis", "max healthcare",
    ],
    "finance": [
        "lic", "hdfc", "sbi", "icici", "bajaj finserv", "policybazaar", "zerodha",
        "groww", "upstox", "angel one", "5paisa", "cleartax", "et money", "kuvera",
        "emi", "loan", "insurance", "mutual", "sip", "premium", "investment", "cred",
    ],
    "entertainment": [
        "netflix", "prime", "hotstar", "spotify", "jiocinema", "sonyliv", "zee5",
        "voot", "altbalaji", "youtube", "saavn", "gaana", "bookmyshow", "dream11", "mpl",
        "winzo",
    ],
    "atm": [
        "atm", "cash withdrawal", "cash wtdl",
    ],
    "transfer": [
        "upi transfer", "paytm", "phonepe", "google pay", "bhim", "amazon pay",
        "mobikwik", "bank transfer", "neft", "rtgs", "imps",
        "upi", "payu", "acct",
    ],
}

# ── Credit Category Keywords ──────────────────────────────────────────────────
# Credits are modeled SEPARATELY from debits.
# 'other_credit' is the fallback — do NOT add it to the priority order.
CREDIT_PRIORITY = [
    "salary",
    "refund",
    "interest",
    "transfer_in",
]

CREDIT_KEYWORDS: dict[str, list[str]] = {
    "salary":      ["salary", "sal", "payroll", "ctc", "stipend"],
    "refund":      ["refund", "cashback", "reversal", "return", "chargeback"],
    "interest":    ["interest", "int cr", "fd interest", "savings interest"],
    "transfer_in": ["upi transfer", "bank transfer", "neft", "imps", "rtgs", "received", "upi", "acct"],
}

# ── Noise Tokens (stripped from remarks before matching) ─────────────────────
# These tokens add no semantic value to transaction remarks.
# With strict regex merchant aliasing, we can wipe heavy UPI routing noise efficiently.
NOISE_TOKENS: set[str] = {
    "ref", "no", "by", "being", "towards", "payment", "txn", "transaction",
    "cr", "dr", "ac", "a/c", "the", "and", "or", "to", "from",
}

# ── Pipeline Constants ────────────────────────────────────────────────────────
FALLBACK_DEBIT_LABEL  = "uncategorized"
FALLBACK_CREDIT_LABEL = "other_credit"

# Minimum labeling coverage before a warning is raised (0.0 – 1.0)
MIN_COVERAGE_THRESHOLD = 0.40

# Single indisputable source of truth for recurring thresholds
RECURRING_CONFIG = {
    "monthly": {
        "type": "monthly",
        "min_gap": 27,
        "max_gap": 33,
        "var": 10
    },
    "weekly": {
        "type": "weekly",
        "min_gap": 6,
        "max_gap": 8,
        "var": 3
    },
    "biweekly": {
        "type": "biweekly",
        "min_gap": 13,
        "max_gap": 16,
        "var": 5
    },
    "quarterly": {
        "type": "quarterly",
        "min_gap": 85,
        "max_gap": 95,
        "var": 20
    },
    "global": {
        "amount_tolerance": 0.20,
        "min_occurrences": 3,
        "fluctuation_penalty_threshold": 0.10
    }
}

# ===========================================================================
# INSIGHT ENGINE — ML-Based Insight & Tip Configuration
# ===========================================================================

# ── Insight Types (classification targets for the Insight Ranker) ─────────────
INSIGHT_TYPES: list[str] = [
    "spending_spike",      # Anomalous single-transaction spike
    "subscription",        # Recurring charge detected
    "trend_warning",       # Category spending trending upward week-over-week
    "budget_risk",         # Cumulative monthly spend exceeding rolling average
    "no_action",           # Normal transaction — no insight warranted
]

# ── Tip Corpus (human-vetted financial tips selected by the Tip Selector) ─────
# Each tip is tagged with applicable categories and insight types.
# Empty 'categories' list means the tip applies to ANY category.
TIP_CORPUS: dict[str, dict] = {
    # ── Food ──────────────────────────────────────────────────────────────────
    "tip_food_spike_01": {
        "text": "A single ₹500 meal substitution per week could save ~₹2,000/month.",
        "categories": ["food"],
        "insights": ["spending_spike"],
    },
    "tip_food_spike_02": {
        "text": "This food expense is significantly above your average. "
                "Consider splitting large orders or using offers/coupons.",
        "categories": ["food"],
        "insights": ["spending_spike"],
    },
    "tip_food_trend_01": {
        "text": "Your food spending has been rising week over week. "
                "Try batch-cooking on weekends to reduce delivery dependency.",
        "categories": ["food"],
        "insights": ["trend_warning", "budget_risk"],
    },
    "tip_food_sub_01": {
        "text": "You have an active food delivery subscription. "
                "Verify it's being used enough to justify the cost.",
        "categories": ["food"],
        "insights": ["subscription"],
    },
    # ── Shopping ──────────────────────────────────────────────────────────────
    "tip_shop_spike_01": {
        "text": "Consider a 24-hour cooling-off rule before purchases over ₹2,000.",
        "categories": ["shopping"],
        "insights": ["spending_spike"],
    },
    "tip_shop_spike_02": {
        "text": "Check if this item is available at a lower price on "
                "a competing platform before completing the purchase.",
        "categories": ["shopping"],
        "insights": ["spending_spike"],
    },
    "tip_shop_trend_01": {
        "text": "Your shopping spend is trending upward this month. "
                "Set a weekly discretionary cap to stay on track.",
        "categories": ["shopping"],
        "insights": ["trend_warning", "budget_risk"],
    },
    "tip_shop_sub_01": {
        "text": "Recurring shopping charge detected. Verify this isn't "
                "an unwanted auto-renewal or subscribe-and-save order.",
        "categories": ["shopping"],
        "insights": ["subscription"],
    },
    # ── Transport ─────────────────────────────────────────────────────────────
    "tip_transport_spike_01": {
        "text": "This ride/travel expense is unusually high. Consider "
                "carpooling, public transit, or booking in advance for discounts.",
        "categories": ["transport"],
        "insights": ["spending_spike"],
    },
    "tip_transport_trend_01": {
        "text": "Your transport costs are climbing. Consider a monthly "
                "pass or switching to two-wheelers for short commutes.",
        "categories": ["transport"],
        "insights": ["trend_warning", "budget_risk"],
    },
    "tip_transport_sub_01": {
        "text": "Recurring fuel/transport charges detected. Track your "
                "mileage to check if route optimisation could cut costs.",
        "categories": ["transport"],
        "insights": ["subscription"],
    },
    # ── Utilities ─────────────────────────────────────────────────────────────
    "tip_util_spike_01": {
        "text": "Utility bill spike detected. Check for unusual usage "
                "or billing errors before the next cycle.",
        "categories": ["utilities"],
        "insights": ["spending_spike"],
    },
    "tip_util_trend_01": {
        "text": "Utility bills trending upward. Check for energy leaks, "
                "standby appliances, or seasonal AC usage.",
        "categories": ["utilities"],
        "insights": ["trend_warning", "budget_risk"],
    },
    "tip_util_sub_01": {
        "text": "Recurring utility bill identified. Consider switching "
                "to budget billing for predictable monthly charges.",
        "categories": ["utilities"],
        "insights": ["subscription"],
    },
    # ── Entertainment ─────────────────────────────────────────────────────────
    "tip_ent_spike_01": {
        "text": "Unusual entertainment expense. Check if this was an "
                "accidental in-app purchase or auto-renewal.",
        "categories": ["entertainment"],
        "insights": ["spending_spike"],
    },
    "tip_ent_trend_01": {
        "text": "Entertainment spending is rising. Audit your active "
                "subscriptions — unused ones drain ₹500–1,000/month silently.",
        "categories": ["entertainment"],
        "insights": ["trend_warning", "budget_risk"],
    },
    "tip_ent_sub_01": {
        "text": "Active streaming/entertainment subscription detected. "
                "Check if you've used it in the last 30 days.",
        "categories": ["entertainment"],
        "insights": ["subscription"],
    },
    # ── Finance ───────────────────────────────────────────────────────────────
    "tip_fin_spike_01": {
        "text": "Unexpected financial charge detected. Verify this isn't "
                "a penalty, late fee, or missed EMI payment.",
        "categories": ["finance"],
        "insights": ["spending_spike"],
    },
    "tip_fin_trend_01": {
        "text": "Financial outflows are increasing. Review outstanding "
                "loans and consider prepaying high-interest debt first.",
        "categories": ["finance"],
        "insights": ["trend_warning", "budget_risk"],
    },
    "tip_fin_sub_01": {
        "text": "Recurring EMI/insurance premium identified. Ensure "
                "auto-debit is linked to a funded account to avoid bounce charges.",
        "categories": ["finance"],
        "insights": ["subscription"],
    },
    # ── Health ────────────────────────────────────────────────────────────────
    "tip_health_spike_01": {
        "text": "Significant health expense detected. Check if this is "
                "claimable under your health insurance policy.",
        "categories": ["health"],
        "insights": ["spending_spike"],
    },
    "tip_health_trend_01": {
        "text": "Health-related spending is trending up. Consider "
                "preventive health check-ups to catch issues early.",
        "categories": ["health"],
        "insights": ["trend_warning", "budget_risk"],
    },
    "tip_health_sub_01": {
        "text": "Recurring pharmacy/health charge detected. Ask your "
                "doctor about generic alternatives for regular medications.",
        "categories": ["health"],
        "insights": ["subscription"],
    },
    # ── ATM ───────────────────────────────────────────────────────────────────
    "tip_atm_spike_01": {
        "text": "Large cash withdrawal detected. ATM withdrawals are "
                "harder to track — consider using UPI for spending visibility.",
        "categories": ["atm"],
        "insights": ["spending_spike"],
    },
    "tip_atm_trend_01": {
        "text": "Cash withdrawals are increasing. Untracked cash spending "
                "is the #1 budget leak — try going cashless for a week.",
        "categories": ["atm"],
        "insights": ["trend_warning", "budget_risk"],
    },
    # ── Transfer ──────────────────────────────────────────────────────────────
    "tip_transfer_spike_01": {
        "text": "Unusually large transfer detected. Verify the recipient "
                "and ensure this wasn't an error or unauthorised transaction.",
        "categories": ["transfer"],
        "insights": ["spending_spike"],
    },
    "tip_transfer_trend_01": {
        "text": "Outgoing transfers trending up. Review if recurring "
                "transfers can be reduced or consolidated.",
        "categories": ["transfer"],
        "insights": ["trend_warning", "budget_risk"],
    },
    # ── Generic (category-agnostic) ───────────────────────────────────────────
    "tip_generic_spike_01": {
        "text": "This transaction is significantly above your normal "
                "spending pattern. Review to ensure it was intentional.",
        "categories": [],
        "insights": ["spending_spike"],
    },
    "tip_generic_trend_01": {
        "text": "Spending in this category is trending upward. Consider "
                "setting a monthly category budget to stay on track.",
        "categories": [],
        "insights": ["trend_warning"],
    },
    "tip_generic_budget_01": {
        "text": "Your cumulative spending this month is outpacing your "
                "historical average. Review non-essential expenses.",
        "categories": [],
        "insights": ["budget_risk"],
    },
    "tip_generic_sub_01": {
        "text": "Recurring charge identified. Periodically review all "
                "subscriptions to cancel unused services.",
        "categories": [],
        "insights": ["subscription"],
    },
}

# ── Insight Templates (multiple phrasings per insight type for variety) ───────
INSIGHT_TEMPLATES: dict[str, list[str]] = {
    "spending_spike": [
        "Unusual {category} expense at '{merchant}' on {date} (₹{amount:.2f}). "
        "{pct:.1f}% above your baseline.",
        "Spending alert: ₹{amount:.2f} at '{merchant}' ({category}) on {date} "
        "exceeds your normal pattern by {pct:.1f}%.",
        "Heads up — '{merchant}' charged ₹{amount:.2f} for {category} on {date}, "
        "which is {pct:.1f}% higher than expected.",
    ],
    "subscription": [
        "Subscription identified: '{merchant}' charges ~₹{amount:.2f} {frequency}.",
        "Recurring charge detected: '{merchant}' bills ~₹{amount:.2f} {frequency}.",
        "Ongoing {frequency} subscription: '{merchant}' at ~₹{amount:.2f}.",
    ],
    "trend_warning": [
        "Your {category} spending has been rising over the last few weeks.",
        "Trend alert: {category} expenses are climbing week-over-week.",
    ],
    "budget_risk": [
        "Cumulative {category} spend this month exceeds your rolling average.",
        "Budget watch: {category} spending is running above historical norms.",
    ],
}


def lookup_matching_tip_ids(category: str, insight_type: str) -> list[str]:
    """
    2-pass TIP_CORPUS lookup: category-specific first, then generic.

    This is the canonical lookup used by both insight_generator._select_tip
    and training_data_generator._find_best_tip. Centralised here to avoid
    duplicate iteration logic.

    Returns:
        List of matching tip_ids (may be empty).
    """
    # Pass 1: category-specific match
    specific = [
        tid for tid, tip in TIP_CORPUS.items()
        if category in tip["categories"] and insight_type in tip["insights"]
    ]
    if specific:
        return specific

    # Pass 2: generic match (empty categories list)
    generic = [
        tid for tid, tip in TIP_CORPUS.items()
        if len(tip["categories"]) == 0 and insight_type in tip["insights"]
    ]
    return generic

# ===========================================================================
# KNOWN PERSONS & SELF ACCOUNTS (Exclusion from spend intelligence)
# ===========================================================================

KNOWN_PERSONS: dict[str, dict] = {
    # User populates. Empty by default = feature disabled.
    # "Mom": {
    #     "names": ["sujata devi", "sujata"],
    #     "upi_ids": ["sujata@ybl", "9876543210@paytm"],
    # },
    "Subho" : {
        "names" : ["subhadeep saha","subhadeep","subhadeepsaha","mrsubhadeep"],
        "upi_ids" : []
    }
}


SELF_ACCOUNTS: dict[str, dict] = {
    # User populates. Empty by default = feature disabled.
    # "HDFC_Savings": {
    #     "names": [],
    #     "account_fragments": ["50100"],
    #     "upi_ids": ["myupi@hdfcbank"],
    # },
}

# Matching Configuration
KNOWN_PERSON_MATCH_THRESHOLD = 2
CONCAT_MIN_LENGTH = 8
CONCAT_PARTIAL_MIN_LENGTH = 4
MIN_SPEND_TRANSACTIONS_FOR_ML = 30

# CRITICAL: _MERCHANT_INDICATOR_TOKENS and _MERCHANT_SUFFIXES affect
# classification correctness, not just heuristics. They are behavioral
# dependencies. Removing tokens will cause misclassification.
# See known_persons.py for usage.

# Personal pattern detection (SEPARATE from RECURRING_CONFIG)
PERSONAL_RECURRING_CONFIG: dict[str, dict] = {
    "monthly": {
        "type": "monthly",
        "min_gap": 22,      # RECURRING_CONFIG uses 27
        "max_gap": 40,      # RECURRING_CONFIG uses 33
        "var": 18,          # RECURRING_CONFIG uses 10
    },
    "weekly": {
        "type": "weekly",
        "min_gap": 5,       # RECURRING_CONFIG uses 6
        "max_gap": 10,      # RECURRING_CONFIG uses 8
        "var": 5,           # RECURRING_CONFIG uses 3
    },
    "global": {
        "amount_tolerance": 0.30,      # RECURRING_CONFIG uses 0.20
        "min_occurrences": 3,          # same
        "fluctuation_penalty_threshold": 0.20,  # RECURRING_CONFIG uses 0.10
    }
}
