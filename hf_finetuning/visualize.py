import json
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Dict

# Load the data from trainer_state.json
def load_data(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Extract relevant data for visualization
def extract_data(log_history: List[Dict]) -> Dict:
    epochs = [entry['epoch'] for entry in log_history if 'epoch' in entry]
    loss = [entry['loss'] for entry in log_history if 'loss' in entry]
    eval_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
    learning_rate = [entry['learning_rate'] for entry in log_history if 'learning_rate' in entry]
    grad_norm = [entry['grad_norm'] for entry in log_history if 'grad_norm' in entry]
    eval_accuracy = [entry['eval_accuracy'] for entry in log_history if 'eval_accuracy' in entry]

    return {
        'epochs': epochs,
        'loss': loss,
        'eval_loss': eval_loss,
        'learning_rate': learning_rate,
        'grad_norm': grad_norm,
        'eval_accuracy': eval_accuracy,
    }

# Create the visualizations
def create_plots(data: Dict):
    epochs = data['epochs']

    # Create subplots: 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=("Training & Evaluation Loss", "Gradient Norm & Learning Rate")
    )

    # Plot Training and Evaluation Loss
    fig.add_trace(
        go.Scatter(
            x=epochs, y=data['loss'], 
            mode='lines+markers', name='Training Loss', 
            marker=dict(color='blue')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=data['eval_loss'], 
            mode='lines+markers', name='Evaluation Loss', 
            marker=dict(color='red')
        ),
        row=1, col=1
    )

    # Plot Gradient Norm and Learning Rate
    fig.add_trace(
        go.Scatter(
            x=epochs, y=data['grad_norm'], 
            mode='lines+markers', name='Gradient Norm', 
            marker=dict(color='green')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=data['learning_rate'], 
            mode='lines+markers', name='Learning Rate', 
            marker=dict(color='orange')
        ),
        row=2, col=1
    )

    # Update layout for the plots
    fig.update_layout(
        title_text='Training Progress Overview',
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            x=1.05, y=1, 
            traceorder='normal', 
            bordercolor='black', 
            borderwidth=1,
            bgcolor='white'
        ),
        title_font=dict(size=20, color='black'),
        plot_bgcolor='white',
    )

    # Update X and Y axis properties
    fig.update_xaxes(
        title_text="Epochs", 
        linecolor='black', 
        showgrid=True, 
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        title_text="Loss", 
        linecolor='black', 
        showgrid=True, 
        gridcolor='lightgrey', 
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Gradient Norm & Learning Rate", 
        linecolor='black', 
        showgrid=True, 
        gridcolor='lightgrey', 
        row=2, col=1
    )

    # Add descriptive text annotations
    eval_accuracy = data['eval_accuracy']
    if any(eval_accuracy):
        fig.add_annotation(
            dict(
                x=epochs[-1],
                y=eval_accuracy[-1],
                text=f"Final Eval Accuracy: {eval_accuracy[-1]*100:.2f}%",
                xref="x1", yref="y1",
                showarrow=True,
                arrowhead=2,
                ax=-40, ay=-40
            )
        )

    # Save the plot as an HTML file
    fig.write_html("training_progress_overview.html")

def main(file_path):
    
    trainer_state = load_data(file_path)
    
    # Extract data from log history
    log_history = trainer_state.get('log_history', [])
    if log_history:
        data = extract_data(log_history)
        
        # Create and save the plots
        create_plots(data)
    else:
        print("Log history is empty or not found.")

if __name__ == "__main__":
    import sys
    try:
        yaml_config_file = sys.argv[1]
    except IndexError:
        raise ValueError("Please provide the path to the path training_state.json")

    main(yaml_config_file)
