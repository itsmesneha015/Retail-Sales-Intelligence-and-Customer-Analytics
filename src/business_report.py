from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
import pandas as pd
import sys
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =================================
# Register Cross-Platform Unicode Font
# =================================

# Windows: use Nirmala UI when available.
# Streamlit Cloud (Linux): fall back to DejaVu Sans.
font_candidates = [
    ("Nirmala", r"C:\Windows\Fonts\Nirmala.ttc"),
    ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]

PDF_FONT_NAME = "Helvetica"

for _font_name, _font_path in font_candidates:
    if os.path.exists(_font_path):
        try:
            pdfmetrics.registerFont(TTFont(_font_name, _font_path))
            PDF_FONT_NAME = _font_name
            break
        except Exception:
            pass


# =================================
# Business Report Generator
# =================================

print("================================")
print("Business Report Generator")
print("================================")


# =================================
# Report Language
# =================================

report_language = "English"

if len(sys.argv) > 1:
    report_language = sys.argv[1]


# =================================
# Load Project Data
# =================================

sales_df = pd.read_csv(
    "data/processed/featured_sales.csv"
)

seasonal_df = pd.read_csv(
    "data/processed/seasonal_demand_predictions.csv"
)

top_seasonal_df = pd.read_csv(
    "data/processed/future_top_category_by_season.csv"
)

customer_df = pd.read_csv(
    "data/processed/customer_entry_exit.csv"
)

dwell_df = pd.read_csv(
    "data/processed/customer_dwell_time.csv"
)

zone_df = pd.read_csv(
    "data/processed/customer_zone_analytics.csv"
)

print("\nProject data loaded successfully!")


# =================================
# Calculate Business Metrics
# =================================

total_sales = sales_df["Sales"].sum()

total_quantity = sales_df["Quantity"].sum()

total_entered = customer_df["Total_Entered"].max()

total_exited = customer_df["Total_Exited"].max()

peak_occupancy = customer_df[
    "Current_Customers"
].max()

average_dwell = dwell_df[
    "Dwell_Time_Seconds"
].mean()


# =================================
# Zone Analytics
# =================================

zone_averages = {
    "Zone 1 - Left": zone_df["Zone_1_Left"].mean(),
    "Zone 2 - Center": zone_df["Zone_2_Center"].mean(),
    "Zone 3 - Right": zone_df["Zone_3_Right"].mean()
}

busiest_zone = max(
    zone_averages,
    key=zone_averages.get
)

busiest_zone_value = zone_averages[
    busiest_zone
]


# =================================
# Highest Predicted Demand
# =================================

highest_demand = seasonal_df.loc[
    seasonal_df["Predicted_Quantity"].idxmax()
]

highest_category = highest_demand[
    "Category"
]

highest_season = highest_demand[
    "Season"
]

highest_quantity = highest_demand[
    "Predicted_Quantity"
]


# =================================
# Report Text by Language
# =================================

if report_language == "Kannada":

    report_title = "ಚಿಲ್ಲರೆ ವ್ಯಾಪಾರ ಬುದ್ಧಿವಂತಿಕೆ ವರದಿ"

    report_subtitle = (
        "ಮಾರಾಟ, ಗ್ರಾಹಕ ಮತ್ತು ಬೇಡಿಕೆ ವಿಶ್ಲೇಷಣೆ"
    )

    sales_heading = "1. ಮಾರಾಟದ ಸಾರಾಂಶ"

    seasonal_heading = (
        "2. ಋತುಮಾನ ಬೇಡಿಕೆ ಮುನ್ಸೂಚನೆ"
    )

    customer_heading = (
        "3. ಗ್ರಾಹಕರ ವಿಶ್ಲೇಷಣೆ"
    )

    zone_heading = (
        "4. ಅಂಗಡಿ ವಲಯ ವಿಶ್ಲೇಷಣೆ"
    )

    recommendation_heading = (
        "5. ವ್ಯಾಪಾರ ಶಿಫಾರಸು"
    )

    metric_header = "ಮಾಪನ"

    value_header = "ಮೌಲ್ಯ"

    total_sales_text = "ಒಟ್ಟು ಮಾರಾಟ"

    total_quantity_text = (
        "ಮಾರಾಟವಾದ ಒಟ್ಟು ಪ್ರಮಾಣ"
    )

    highest_demand_text = (
        f"ಅತಿ ಹೆಚ್ಚು ನಿರೀಕ್ಷಿತ ಬೇಡಿಕೆ: "
        f"{highest_category} - {highest_season}, "
        f"ಸುಮಾರು {int(highest_quantity)} ಘಟಕಗಳು."
    )

    recommendation_text = (
        f"ಹೆಚ್ಚಿನ ಬೇಡಿಕೆ ನಿರೀಕ್ಷೆಯಿರುವ "
        f"{highest_season} ಸಮಯದಲ್ಲಿ "
        f"{highest_category} ಉತ್ಪನ್ನಗಳ ಸಾಕಷ್ಟು "
        f"ಸ್ಟಾಕ್ ಇಟ್ಟುಕೊಳ್ಳಲು ಪರಿಗಣಿಸಿ. "
        f"{busiest_zone} ನಲ್ಲಿ ಗ್ರಾಹಕರ "
        f"ಚಟುವಟಿಕೆಯನ್ನು ಮೇಲ್ವಿಚಾರಣೆ "
        f"ಮಾಡುವುದು ಉತ್ತಮ."
    )

    seasonal_header_season = "ಋತು"

    seasonal_header_category = "ವರ್ಗ"

    seasonal_header_quantity = (
        "ನಿರೀಕ್ಷಿತ ಪ್ರಮಾಣ"
    )

    zone_header = "ಅಂಗಡಿ ವಲಯ"

    zone_average_header = (
        "ಸರಾಸರಿ ಗ್ರಾಹಕರು"
    )

    customer_entered_text = (
        "ಪ್ರವೇಶಿಸಿದ ಗ್ರಾಹಕರು"
    )

    customer_exited_text = (
        "ಹೊರಹೋದ ಗ್ರಾಹಕರು"
    )

    peak_customers_text = (
        "ಗರಿಷ್ಠ ಗ್ರಾಹಕರು"
    )

    average_visible_text = (
        "ಸರಾಸರಿ ಅಂದಾಜು ಗೋಚರ ಸಮಯ"
    )

    highest_activity_text = (
        f"ಹೆಚ್ಚಿನ ಗ್ರಾಹಕರ ಚಟುವಟಿಕೆ "
        f"{busiest_zone} ನಲ್ಲಿ ಕಂಡುಬಂದಿದೆ. "
        f"ಸರಾಸರಿ ಗ್ರಾಹಕರು "
        f"{busiest_zone_value:.2f}."
    )


elif report_language == "Hindi":

    report_title = (
        "रिटेल बिजनेस इंटेलिजेंस रिपोर्ट"
    )

    report_subtitle = (
        "बिक्री, ग्राहक और मांग विश्लेषण"
    )

    sales_heading = "1. बिक्री सारांश"

    seasonal_heading = (
        "2. अनुमानित मौसमी मांग"
    )

    customer_heading = (
        "3. ग्राहक विश्लेषण"
    )

    zone_heading = (
        "4. स्टोर ज़ोन विश्लेषण"
    )

    recommendation_heading = (
        "5. व्यापार सुझाव"
    )

    metric_header = "मापदंड"

    value_header = "मान"

    total_sales_text = "कुल बिक्री"

    total_quantity_text = (
        "बेची गई कुल मात्रा"
    )

    highest_demand_text = (
        f"सबसे अधिक अनुमानित मांग: "
        f"{highest_category} - {highest_season}, "
        f"लगभग {int(highest_quantity)} यूनिट।"
    )

    recommendation_text = (
        f"उच्च मांग वाले {highest_season} "
        f"समय में {highest_category} का "
        f"पर्याप्त स्टॉक रखने पर विचार करें। "
        f"{busiest_zone} में ग्राहक गतिविधि "
        f"की निगरानी करना उपयोगी होगा।"
    )

    seasonal_header_season = "मौसम"

    seasonal_header_category = "श्रेणी"

    seasonal_header_quantity = (
        "अनुमानित मात्रा"
    )

    zone_header = "स्टोर ज़ोन"

    zone_average_header = (
        "औसत ग्राहक"
    )

    customer_entered_text = (
        "प्रवेश करने वाले ग्राहक"
    )

    customer_exited_text = (
        "बाहर जाने वाले ग्राहक"
    )

    peak_customers_text = (
        "अधिकतम ग्राहक"
    )

    average_visible_text = (
        "औसत अनुमानित दृश्य समय"
    )

    highest_activity_text = (
        f"सबसे अधिक ग्राहक गतिविधि "
        f"{busiest_zone} में देखी गई। "
        f"औसत ग्राहक {busiest_zone_value:.2f} हैं।"
    )


else:

    report_title = (
        "Retail Business Intelligence Report"
    )

    report_subtitle = (
        "Sales, Customer and Demand Analysis"
    )

    sales_heading = "1. Sales Summary"

    seasonal_heading = (
        "2. Predicted Seasonal Demand"
    )

    customer_heading = (
        "3. Customer Analytics"
    )

    zone_heading = (
        "4. Store Zone Analytics"
    )

    recommendation_heading = (
        "5. Business Recommendation"
    )

    metric_header = "Metric"

    value_header = "Value"

    total_sales_text = "Total Sales"

    total_quantity_text = (
        "Total Quantity Sold"
    )

    highest_demand_text = (
        f"Highest predicted demand: "
        f"{highest_category} during "
        f"{highest_season}, with approximately "
        f"{int(highest_quantity)} units."
    )

    recommendation_text = (
        f"Consider maintaining sufficient stock "
        f"of {highest_category} during the predicted "
        f"high-demand {highest_season} period. "
        f"Customer activity should also be monitored "
        f"in {busiest_zone} to support effective "
        f"store management."
    )

    seasonal_header_season = "Season"

    seasonal_header_category = "Category"

    seasonal_header_quantity = (
        "Predicted Quantity"
    )

    zone_header = "Store Zone"

    zone_average_header = (
        "Average Customers"
    )

    customer_entered_text = (
        "Customers Entered"
    )

    customer_exited_text = (
        "Customers Exited"
    )

    peak_customers_text = (
        "Peak Customers"
    )

    average_visible_text = (
        "Average Estimated Visible Time"
    )

    highest_activity_text = (
        f"Highest customer activity was observed "
        f"in {busiest_zone}, with an average of "
        f"{busiest_zone_value:.2f} customers."
    )


# =================================
# Output PDF
# =================================

output_file = (
    "reports/retail_business_report.pdf"
)


doc = SimpleDocTemplate(
    output_file,
    pagesize=A4
)


# =================================
# Styles
# =================================

styles = getSampleStyleSheet()


title_style = styles["Title"]
title_style.alignment = TA_CENTER
title_style.fontName = PDF_FONT_NAME


heading_style = styles["Heading2"]
heading_style.fontName = PDF_FONT_NAME


normal_style = styles["BodyText"]
normal_style.fontName = PDF_FONT_NAME


# =================================
# PDF Story
# =================================

story = []


# =================================
# Title
# =================================

story.append(
    Paragraph(
        report_title,
        title_style
    )
)

story.append(
    Spacer(1, 20)
)


# =================================
# Subtitle
# =================================

story.append(
    Paragraph(
        report_subtitle,
        normal_style
    )
)

story.append(
    Spacer(1, 20)
)


# =================================
# Sales Summary
# =================================

story.append(
    Paragraph(
        sales_heading,
        heading_style
    )
)


sales_table = Table([
    [
        metric_header,
        value_header
    ],
    [
        total_sales_text,
        f"{total_sales:.2f}"
    ],
    [
        total_quantity_text,
        f"{total_quantity}"
    ]
])


sales_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, -1),
            PDF_FONT_NAME
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            6
        )
    ])
)


story.append(sales_table)

story.append(
    Spacer(1, 20)
)


# =================================
# Seasonal Demand
# =================================

story.append(
    Paragraph(
        seasonal_heading,
        heading_style
    )
)


seasonal_table_data = [
    [
        seasonal_header_season,
        seasonal_header_category,
        seasonal_header_quantity
    ]
]


for _, row in top_seasonal_df.iterrows():

    seasonal_table_data.append([
        row["Season"],
        row["Category"],
        str(
            int(
                row["Predicted_Quantity"]
            )
        )
    ])


seasonal_table = Table(
    seasonal_table_data
)


seasonal_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, -1),
            PDF_FONT_NAME
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            6
        )
    ])
)


story.append(
    seasonal_table
)

story.append(
    Spacer(1, 10)
)


story.append(
    Paragraph(
        highest_demand_text,
        normal_style
    )
)

story.append(
    Spacer(1, 20)
)


# =================================
# Customer Analytics
# =================================

story.append(
    Paragraph(
        customer_heading,
        heading_style
    )
)


customer_table = Table([
    [
        metric_header,
        value_header
    ],
    [
        customer_entered_text,
        str(total_entered)
    ],
    [
        customer_exited_text,
        str(total_exited)
    ],
    [
        peak_customers_text,
        str(peak_occupancy)
    ],
    [
        average_visible_text,
        f"{average_dwell:.2f} seconds"
    ]
])


customer_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, -1),
            PDF_FONT_NAME
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            6
        )
    ])
)


story.append(customer_table)

story.append(
    Spacer(1, 20)
)


# =================================
# Store Zone Analytics
# =================================

story.append(
    Paragraph(
        zone_heading,
        heading_style
    )
)


zone_table_data = [
    [
        zone_header,
        zone_average_header
    ]
]


for zone, value in zone_averages.items():

    zone_table_data.append([
        zone,
        f"{value:.2f}"
    ])


zone_table = Table(
    zone_table_data
)


zone_table.setStyle(
    TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, -1),
            PDF_FONT_NAME
        ),
        (
            "PADDING",
            (0, 0),
            (-1, -1),
            6
        )
    ])
)


story.append(zone_table)

story.append(
    Spacer(1, 10)
)


story.append(
    Paragraph(
        highest_activity_text,
        normal_style
    )
)

story.append(
    Spacer(1, 20)
)


# =================================
# Business Recommendation
# =================================

story.append(
    Paragraph(
        recommendation_heading,
        heading_style
    )
)


story.append(
    Paragraph(
        recommendation_text,
        normal_style
    )
)


# =================================
# Generate PDF
# =================================

doc.build(story)


print(
    "\nBusiness report generated successfully!"
)

print(
    f"Report saved to: {output_file}"
)