from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to avoid threading issues with Tkinter
import matplotlib.pyplot as plt
import io
import base64
import os
from model import (
    get_recommendation, train_model, load_model, predict_next_numbers,
    reset_model, train_color_model, load_color_model, predict_color,
    train_parity_model, load_parity_model, predict_parity,
    train_range_model, load_range_model, predict_range
)

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATA_FILE = 'roulette_data.csv'
BETS_FILE = 'bets_data.csv'

# Function to check and create the CSV file if it doesn't exist
def check_and_create_data_file():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=['Number'])
        df.to_csv(DATA_FILE, index=False)

def check_and_create_bets_file():
    if not os.path.exists(BETS_FILE):
        df = pd.DataFrame(columns=['Bet', 'Result', 'Amount', 'Net_Gain'])
        df.to_csv(BETS_FILE, index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    check_and_create_data_file()
    check_and_create_bets_file()
    
    if request.method == 'POST':
        number = request.form.get('number')
        if number:
            try:
                number = int(number)
                if 0 <= number <= 36:
                    data = pd.read_csv(DATA_FILE)
                    new_row = pd.DataFrame({'Number': [number]})
                    data = pd.concat([data, new_row], ignore_index=True)
                    data.to_csv(DATA_FILE, index=False)
                    flash(f'Number {number} added successfully!', 'success')
                    
                    model, max_len = load_model()
                    min_numbers = 20  # Minimum number of entries needed to train the model
                    
                    if len(data) >= min_numbers and model:
                        recommendations = predict_next_numbers(model, data['Number'].tolist(), max_len, num_predictions=10)
                        if number in recommendations:
                            result = 'Win'
                            net_gain = 36 * int(session.get('bet_amount', 1)) - 10 * int(session.get('bet_amount', 1))
                        else:
                            result = 'Loss'
                            net_gain = -10 * int(session.get('bet_amount', 1))
                        
                        bet_row = pd.DataFrame({
                            'Bet': [number],
                            'Result': [result],
                            'Amount': [session.get('bet_amount', 1)],
                            'Net_Gain': [net_gain]
                        })
                        
                        bets_data = pd.read_csv(BETS_FILE)
                        bets_data = pd.concat([bets_data, bet_row], ignore_index=True)
                        bets_data.to_csv(BETS_FILE, index=False)
                        
                    else:
                        flash('Not enough data to make recommendations', 'danger')
                    
                else:
                    flash('Number must be between 0 and 36', 'danger')
            except ValueError:
                flash('Please enter a valid number', 'danger')
        return redirect(url_for('index'))
    
    data = pd.read_csv(DATA_FILE)
    model, max_len = load_model()
    color_model, color_label_encoder = load_color_model()
    parity_model, parity_label_encoder = load_parity_model()
    range_model, range_label_encoder = load_range_model()
    min_numbers = 20  # Minimum number of entries needed to train the model
    
    if len(data) >= min_numbers and model:
        recommendations = sorted(predict_next_numbers(model, data['Number'].tolist(), max_len, num_predictions=10))
        color_recommendation = predict_color(color_model, data['Number'].tolist(), max_len, color_label_encoder)
        parity_recommendation = predict_parity(parity_model, data['Number'].tolist(), max_len, parity_label_encoder)
        range_recommendation = sorted(predict_range(range_model, data['Number'].tolist(), max_len, range_label_encoder, num_predictions=2))
    else:
        recommendations = ["Not enough data to make recommendations"] if len(data) < min_numbers else ["Model not trained yet"]
        color_recommendation = "Not enough data to recommend color"
        parity_recommendation = "Not enough data to recommend parity"
        range_recommendation = ["Not enough data to recommend range"]
    
    plot_url = generate_plot(data['Number'].tolist())
    
    bets_data = pd.read_csv(BETS_FILE)
    total_bets = len(bets_data)
    total_winnings = bets_data[bets_data['Result'] == 'Win']['Net_Gain'].sum()
    total_losses = bets_data[bets_data['Result'] == 'Loss']['Net_Gain'].sum()
    
    return render_template(
        'index.html', 
        numbers=data['Number'].tolist(), 
        recommendations=recommendations, 
        color_recommendation=color_recommendation,
        parity_recommendation=parity_recommendation,
        range_recommendation=range_recommendation,
        plot_url=plot_url,
        bets_data=bets_data.to_dict(orient='records'),
        total_bets=total_bets,
        total_winnings=total_winnings,
        total_losses=total_losses,
        bet_amount=session.get('bet_amount', 1)
    )

@app.route('/train', methods=['POST'])
def train():
    data = pd.read_csv(DATA_FILE)
    train_model(data['Number'].tolist())
    train_color_model(data['Number'].tolist())
    train_parity_model(data['Number'].tolist())
    train_range_model(data['Number'].tolist())
    flash('Model trained successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    reset_model()
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(BETS_FILE):
        os.remove(BETS_FILE)
    session.pop('bet_amount', None)
    flash('Model and data reset successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/set_bet_amount', methods=['POST'])
def set_bet_amount():
    bet_amount = request.form.get('bet_amount')
    if bet_amount and bet_amount.isdigit() and int(bet_amount) > 0:
        session['bet_amount'] = int(bet_amount)
        flash(f'Bet amount set to {bet_amount}', 'success')
    else:
        flash('Please enter a valid bet amount', 'danger')
    return redirect(url_for('index'))

def generate_plot(numbers):
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(numbers, bins=range(0, 38), edgecolor='black')
    ax.set_title('Number Frequency')
    ax.set_xlabel('Number')
    ax.set_ylabel('Frequency')
    
    # Add frequency labels above each bar
    for count, bin, patch in zip(counts, bins, patches):
        height = patch.get_height()
        ax.annotate(f'{int(count)}', xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

if __name__ == "__main__":
    app.run(debug=True)
