# Scaledux
Built a Startup Evaluation Engine that scores each startup from 0 to 100, like a credit score, based on their business health.

1. Loaded the Dataset
   Read the CSV file containing details of 100 startups.
   Each startup had features like:
     Team experience
     Market size
     Number of users
     Burn rate (monthly expenses)
     Funds raised
     Company valuation

2. Preprocessed the Data
   Used Min-Max normalization to scale every numeric column to a 0 to 1 range.
  
  Why? So that one feature (like valuation) doesn't overpower others just because it has larger numbers.

Important Logic Applied:

  For positive indicators like team experience, market size, users, funding, and valuation, higher is better — so kept them as is.
  
  For burn rate, higher is worse — so inverted it after normalization using 1 - normalized_value.

3. Scoring Logic
  Created a custom score using a weighted average.
  Here’s how decided the weights (importance) of each feature:
    Feature	Weight (%)	Reason
    Team Experience	15%	Strong team = better execution potential
    Market Size	15%	Bigger market = more room to grow
    Monthly Active Users	20%	Shows traction and demand
    Burn Rate (inverted)	10%	Less burn = better efficiency
    Funds Raised	20%	Indicates investor trust and runway
    Valuation	20%	Reflects market confidence and business potential

Then calculated the score out of 100 using this weighted formula.

4. Ranking the Startups
  Sorted all startups by their final score.

  Extracted:

    Top 10 performers → healthiest startups
    
    Bottom 10 performers → underperforming or risky ones

5. Interpreting Scores 
  Startup X (Top Scorer):
  Had high user base, strong team, raised good funds, and efficient burn rate → hence, high score (~95+)
  
  Startup Y (Low Scorer):
  Had low users, high burn rate, little funding → scored low (~30)
6. ML Model 
    Trained a basic regression model using to predict the health score of new startups automatically.

7. Visualization
  Also added visual insights:
    Bar chart showing all startups' scores
   Correlation heatmap to see which features are related
   Histogram to see how scores are spread across all startups
