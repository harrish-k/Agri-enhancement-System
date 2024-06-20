from flask import Flask, render_template, request, redirect, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import time
import csv
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot



app = Flask(__name__)
app.secret_key = 'hi_this_is_harrish_cool!'

@app.route('/check_value_page')
def check_value_page():
    return render_template('recommendation_input.html')

def calculate_percentage_reduction(actual_value, optimal_range_start, optimal_range_end):
    if actual_value < optimal_range_start:
        percentage_reduction = ((optimal_range_end - optimal_range_start) - (actual_value - optimal_range_start)) / (optimal_range_end - optimal_range_start) * 100
    elif actual_value > optimal_range_end:
        percentage_reduction = 100
    else:
        percentage_reduction = 0
    return round(percentage_reduction, 2)

def check_element_levels(nitrogen,phosphorus, sulfur, zinc, iron, manganese, copper, potassium, calcium, magnesium, sodium):
    Crop = session.get('Crop')
    if (Crop == "Rice") :
        normal_ranges = {"Nitrogen (N)" : (150,250),"Phosphorus (P)": (40, 80),"Potassium (K)": (100, 200),
                        "Sulfur (S)": (10, 20),"Zinc (Zn)": (1, 2),"Iron (Fe)": (20, 80),
                        "Manganese (Mn)": (1, 5),"Copper (Cu)": (0.1, 3),"Calcium (Ca)": (400, 1000),
                        "Magnesium (Mg)": (50, 200),"Sodium (Na)": (4, 20) }
    elif (Crop == "Groundnut"):
        normal_ranges = {"Nitrogen (N)" : (40,50),"Phosphorus (P)": (20, 30),"Potassium (K)": (30, 40),
                        "Sulfur (S)": (10, 15),"Zinc (Zn)": (1, 2),"Iron (Fe)": (40, 80),
                        "Manganese (Mn)": (2, 5),"Copper (Cu)": (0.2, 1),"Calcium (Ca)": (1000, 2000),
                        "Magnesium (Mg)": (200, 400),"Sodium (Na)": (0, 20) }
    elif(Crop == "Black Gram"):
        normal_ranges = {"Nitrogen (N)" : (25,35),"Phosphorus (P)": (40, 50),"Potassium (K)": (20, 30),
                        "Sulfur (S)": (10, 15),"Zinc (Zn)": (1, 2),"Iron (Fe)": (40, 80),
                        "Manganese (Mn)": (2, 5),"Copper (Cu)": (0.2, 1.0),"Calcium (Ca)": (1000, 2000),
                        "Magnesium (Mg)": (200, 400),"Sodium (Na)": (0, 20) }
    elif(Crop == "Bengal Gram"):
        normal_ranges = {"Nitrogen (N)" : (25,35),"Phosphorus (P)": (40, 50),"Potassium (K)": (20, 30),
                        "Sulfur (S)": (10, 15),"Zinc (Zn)": (1, 2),"Iron (Fe)": (40, 80),
                        "Manganese (Mn)": (2, 5),"Copper (Cu)": (0.2, 1.0),"Calcium (Ca)": (1000, 2000),
                        "Magnesium (Mg)": (200, 400),"Sodium (Na)": (0, 20) }

    results = []
    total_reduction = 0
    count = 0
    for element, value in zip(normal_ranges.keys(), [nitrogen,phosphorus,potassium, sulfur, zinc, iron, manganese, copper,  calcium, magnesium, sodium]):
        lower_limit, upper_limit = normal_ranges[element]
        percentage_reduction = None  # Initialize percentage_reduction
        if value < lower_limit:
            status = "Below the range"
            if lower_limit != 0:  # Check for division by zero
                percentage_reduction = round(((lower_limit - value) / lower_limit) * 50, 2)
            else:
                percentage_reduction = None
            total_reduction += percentage_reduction
            count += 1
        elif value > upper_limit:
            status = "Above the range"
            if upper_limit != 0:  # Check for division by zero
                percentage_reduction = round(((value - upper_limit) / upper_limit) * 50, 2)
            else:
                percentage_reduction = None
            total_reduction += percentage_reduction
            count += 1
        else:
            status = "Within the range"
        results.append({"Element": element, "Status": status, "Input Value": value, "Percentage Reduction": percentage_reduction})

    if count == 0:
        average_reduction = 0
        message = "No reduction in yield"
    else:
        average_reduction = total_reduction / count
        message = ""

    return results, average_reduction, message

@app.route('/', methods=['GET', 'POST'])
def index():
    with open('fertilizer_recommendation.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        recommendations = [row for row in csv_reader]

    if request.method == 'POST':
        # Your existing code for getting form inputs
        phosphorus = float(request.form['phosphorus'])
        sulfur = float(request.form['sulfur'])
        zinc = float(request.form['zinc'])
        iron = float(request.form['iron'])
        manganese = float(request.form['manganese'])
        copper = float(request.form['copper'])
        potassium = float(request.form['potassium'])
        calcium = float(request.form['calcium'])
        magnesium = float(request.form['magnesium'])
        sodium = float(request.form['sodium'])
        nitrogen = float(request.form['nitrogen'])
        Crop = request.form['Crop']
        session['Crop'] = request.form['Crop']
        
        # Check element levels
        results = check_element_levels(nitrogen,phosphorus, sulfur, zinc, iron, manganese, copper, potassium, calcium, magnesium, sodium)
        
        session['recommendations'] = recommendations

        session['results'] = results[0]

        # Check if results is not None before using it
        session['average_reduction'] = round(results[1], 3) if results is not None else None

        return render_template('recommendations_page.html',Crop = Crop,average_reduction=session.get('average_reduction'), results=results[0], message=results[2], recommendations=recommendations)

    return render_template('hero.html')
    

@app.route('/get_inputs')
def upload_csv():
    # Your logic here
    return render_template('upload_csv.html')


#----------------------------------------------------------- this is the part where predictions get started --------------------------------------------------------------#


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    if not file:
        return redirect(request.url)

    try:
        file.save('uploaded_file.csv')
        dataset = pd.read_csv('uploaded_file.csv')

        districts = dataset['district'].unique().tolist()
        crops = dataset['crop'].unique().tolist()
        seasons = dataset['season'].unique().tolist()
        
        
        time.sleep(9)
        return render_template('prediction_inputs.html', districts=districts, crops=crops, seasons=seasons, crop = session.get('Crop'))
    except pd.errors.ParserError as e:
        return render_template('upload_csv.html', error='Invalid CSV file format. Please upload a valid CSV file.')
    except KeyError as e:
        return render_template('upload_csv.html', error='CSV file does not contain required columns. Please check the format and try again.')

from sklearn.ensemble import RandomForestRegressor

@app.route('/predict', methods=['POST'])
def predict():

    average_reduction = session.get('average_reduction', 0.0)

    try:
        year = int(request.form['year'])
        year_get = int(request.form['year'])
        district = request.form['district']
        crop = session.get('Crop')
        season = request.form['season']
        area = float(request.form['area'])
    except ValueError as e:
        return render_template('prediction_inputs.html', error='Invalid input format. Please enter valid values.')

    dataset = pd.read_csv('uploaded_file.csv')
    y = dataset['production']
    X = dataset[['year', 'district', 'crop', 'season', 'area']]

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    categorical_features = ['district', 'crop', 'season']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)],
        remainder='passthrough')

    random_state = 42
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=random_state))
    ])

    pipeline.fit(X, y_scaled)

    predictions = {}
    predictions_without_reduction = {}  
    if season == 'Both':
        season_types = ['Rabi', 'Kharif']
    else:
        season_types = [season]

    for season_type in season_types:
        season_predictions = []
        season_predictions_without_reduction = []  # Initialize a list for predictions without reduction
        for i in range(6):
            year_pred = year + i
            input_df = pd.DataFrame({'year': [year_pred], 'district': [district], 'crop': [crop], 'season': [season_type], 'area': [area]})

            random_state = 42 + i
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(random_state=random_state))
            ])
            pipeline.fit(X, y_scaled)

            prediction_scaled = pipeline.predict(input_df)
            prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
            season_predictions.append((year_pred, prediction))

            # Calculate prediction without reduction and append to the separate list
            prediction_without_reduction = prediction * (1 - average_reduction / 100)
            season_predictions_without_reduction.append((year_pred, prediction_without_reduction))

        predictions[season_type] = season_predictions
        predictions_without_reduction[season_type] = season_predictions_without_reduction  # Store predictions without reduction in the dictionary
        
    # Create a line plot for predictions without reduction
    line_fig_without_reduction = go.Figure()
    for season_type, season_predictions in predictions_without_reduction.items():
        years = [year for year, _ in season_predictions]
        production = [pred for _, pred in season_predictions]
        line_fig_without_reduction.add_trace(
            go.Scatter(x=years, y=production, mode='lines+markers', name=f'{season_type} (with reduction)',
                    text=[f'{pred:.2f}' for pred in production], hoverinfo='text'))

    line_fig_without_reduction.update_layout(title='Yield Prediction with Reduction (period: 6 Years)'
                                             ,xaxis_title='Years', yaxis_title='Production(kg/ha)Tons')

    # Create a line plot for predictions with reduction
    line_fig_with_reduction = go.Figure()
    for season_type, season_predictions in predictions.items():
        years = [year for year, _ in season_predictions]
        production = [pred for _, pred in season_predictions]
        line_fig_with_reduction.add_trace(
            go.Scatter(x=years, y=production, mode='lines+markers', name=f'{season_type} (without reduction)',
                    text=[f'{pred:.2f}' for pred in production], hoverinfo='text'))

    line_fig_with_reduction.update_layout(title='Yield Prediction without Reduction (period: 6 Years)'
                                          ,xaxis_title='Year', yaxis_title='Production(kg/ha)Tons')

    # Create a bar graph
    bar_fig = go.Figure()
    for season_type in predictions_without_reduction.keys():
        years_season = [year for year, _ in predictions_without_reduction[season_type]]
        bar_fig.add_trace(
            go.Bar(x=years_season, y=[pred for _, pred in predictions_without_reduction[season_type]],
                name=f'{season_type} with reduction', text=[f'{pred:.2f}' for _, pred in predictions_without_reduction[season_type]], hoverinfo='text'))
        years_season = [year for year, _ in predictions[season_type]]
        bar_fig.add_trace(
            go.Bar(x=years_season, y=[pred for _, pred in predictions[season_type]],
                name=f'{season_type} without reduction', text=[f'{pred:.2f}' for _, pred in predictions[season_type]], hoverinfo='text'))
    
    bar_fig.update_layout(title='Yield Prediction Comparison (period: 6 Years)'
                          ,xaxis_title='Year', yaxis_title='Production(kg/ha)Tons')

    production_by_crop_year = dataset.groupby(['year', 'crop'])['production'].sum().reset_index()
    fig1 = px.line(production_by_crop_year, x='year', y='production', color='crop', title='Sum of Production Over the Years for Different Crops')
    production_by_district_year = dataset.groupby(['year', 'district'])['production'].sum().reset_index()
    fig2 = px.line(production_by_district_year, x='year', y='production', color='district', title='Sum of Production With Respect to Districts Over the Years')
    production_by_crop = dataset.groupby('crop')['production'].sum().reset_index()
    fig_pie = go.Figure(data=[go.Pie(labels=production_by_crop['crop'], values=production_by_crop['production'], hole=.5)])
    fig_pie.update_layout(title='Sum of Production by Crop')

    # Save figures as HTML
    line_fig_without_reduction.write_html('static/images/production_prediction_without_reduction.html', auto_open=False)
    line_fig_with_reduction.write_html('static/images/production_prediction_with_reduction.html', auto_open=False)
    bar_fig.write_html('static/images/production_prediction_bar_graph.html', auto_open=False)
    plot(fig_pie, filename='static/images/sum_of_production_by_crop_pie.html', auto_open=False)
    plot(fig2, filename='static/images/sum_of_production_by_districts.html', auto_open=False)
    plot(fig1, filename='static/images/sum_of_production_by_crops.html', auto_open=False)


    return render_template('final_dashboard.html', year=year_get, crop=crop, district=district, area=area, season=season,
                           prediction=round(season_predictions[0][1], 2), yield_prediction=round(season_predictions[0][0], 2) / area,
                           onehec=round(season_predictions[0][0] / area) / 1000, predictions=predictions,
                           predictions_without_reduction=predictions_without_reduction,
                           average_reduction=session.get('average_reduction'),results = session.get('results'),
                           recommendations = session.get('recommendations'))

if __name__ == '__main__':
    app.run(debug=True)
