import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import requests
import json


# Cache for exchange rates to avoid multiple API calls
if 'exchange_rates' not in st.session_state:
    st.session_state['exchange_rates'] = {}
    st.session_state['last_rate_update'] = None

# Function to fetch exchange rates
def get_exchange_rates(base_currency='AUD'):
    """Fetch exchange rates from a public API"""
    # Check if we already have recent rates (less than 1 hour old)
    current_time = datetime.now()
    if (st.session_state['last_rate_update'] is not None and 
        (current_time - st.session_state['last_rate_update']).total_seconds() < 3600 and
        base_currency in st.session_state['exchange_rates']):
        return st.session_state['exchange_rates'][base_currency]
    
    try:
        # Using exchangerate-api.com for free currency conversion
        url = f"https://open.er-api.com/v6/latest/{base_currency}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if data['result'] == 'success':
                # Store the rates in session state
                st.session_state['exchange_rates'][base_currency] = data['rates']
                st.session_state['last_rate_update'] = current_time
                return data['rates']
            else:
                st.error(f"API Error: {data.get('error-type', 'Unknown error')}")
                return None
        else:
            st.error(f"Failed to fetch exchange rates. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching exchange rates: {str(e)}")
        return None

# Add Prophet for forecasting
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
    prophet_available = True
except ImportError:
    prophet_available = False

st.set_page_config(page_title="P2P Trading Analysis", layout="wide")

st.title("P2P Trading Analysis Dashboard")

# Sidebar for file upload and filters
with st.sidebar:
    st.markdown("## 📤 Upload Data")
    
    # Add clear button
    if 'df' in st.session_state and st.button('Clear Current Data'):
        del st.session_state['df']
        st.experimental_rerun()
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df  # Store in session state
        
        # Convert time columns to datetime
        df['Time completed'] = pd.to_datetime(df['Time completed'].str.split(' ').str[0:2].str.join(' '))
        df['Time created'] = pd.to_datetime(df['Time created'].str.split(' ').str[0:2].str.join(' '))
        
        # Date filter settings
        st.markdown("## 📅 Date Filter")
        min_date = df['Time completed'].min().date()
        max_date = df['Time completed'].max().date()
        
        # Default to showing last 30 days if range is large
        default_start = max(min_date, max_date - timedelta(days=30))
        
        start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        # Ensure end_date is not before start_date
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
        
        # Filter data based on date range
        filtered_df = df[(df['Time completed'].dt.date >= start_date) & 
                          (df['Time completed'].dt.date <= end_date)]
        
        st.write(f"Showing data from {start_date} to {end_date}")
        st.write(f"Total trades in selected period: {len(filtered_df)}")
        
        # Status filter
        statuses = df['Status'].unique().tolist()
        selected_statuses = st.multiselect("Filter by Status", statuses, default=statuses)
        
        if selected_statuses:
            filtered_df = filtered_df[filtered_df['Status'].isin(selected_statuses)]
        
        # Currency conversion settings
        st.markdown("## 💱 Currency Settings")
        
        # Add a separator and explanation
        st.markdown("---")
        st.markdown("**🔄 Convert all profit and volume values to your preferred currency**")
        
        # Define common currencies
        common_currencies = {
            'AUD': 'Australian Dollar (AUD)',
            'USD': 'US Dollar (USD)',
            'EUR': 'Euro (EUR)',
            'GBP': 'British Pound (GBP)',
            'JPY': 'Japanese Yen (JPY)',
            'CAD': 'Canadian Dollar (CAD)',
            'CHF': 'Swiss Franc (CHF)',
            'NZD': 'New Zealand Dollar (NZD)',
            'THB': 'Thai Baht (THB)',
            'SGD': 'Singapore Dollar (SGD)',
            'HKD': 'Hong Kong Dollar (HKD)',
            'MYR': 'Malaysian Ringgit (MYR)',
            'INR': 'Indian Rupee (INR)',
            'CNY': 'Chinese Yuan (CNY)'
        }
        
        # Get currency symbol
        currency_symbols = {
            'AUD': 'A$', 'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
            'CAD': 'C$', 'CHF': 'CHF', 'NZD': 'NZ$', 'THB': '฿', 'SGD': 'S$',
            'HKD': 'HK$', 'MYR': 'RM', 'INR': '₹', 'CNY': '¥'
        }
        
        # Default to AUD since that's the original data currency
        display_currency = st.selectbox(
            "Display Currency", 
            options=list(common_currencies.keys()),
            format_func=lambda x: common_currencies[x],
            index=0  # Default to AUD
        )
        
        currency_symbol = currency_symbols.get(display_currency, display_currency)
        
        # Fetch exchange rates if needed
        if display_currency != 'AUD':
            exchange_rates = get_exchange_rates('AUD')
            if not exchange_rates:
                st.warning("Could not fetch exchange rates. Displaying in original currency (AUD).")
                display_currency = 'AUD'
                currency_rate = 1.0
            else:
                currency_rate = exchange_rates.get(display_currency, 1.0)
                st.success(f"Using exchange rate: 1 AUD = {currency_rate} {display_currency}")
        else:
            currency_rate = 1.0
        
        # Store the currency rate in session state to ensure consistency
        st.session_state['currency_rate'] = currency_rate
        st.session_state['display_currency'] = display_currency
        st.session_state['currency_symbol'] = currency_symbol
        
        # ROI calculation settings
        st.markdown("## 📊 ROI Settings")
        
        # Add a separator to make this section more prominent
        st.markdown("---")
        
        # Toggle switch for display mode (currency or percentage) - use radio buttons for more prominence
        if 'show_percentage' not in st.session_state:
            st.session_state['show_percentage'] = False
        
        st.markdown("**📊 Choose how to display profit values:**")
        display_mode = st.radio(
            "",  # Empty label since we're using markdown above
            options=[
                "💰 Currency Values (actual profit amounts)",
                "📈 ROI Percentages (return on investment)"
            ],
            index=0 if not st.session_state['show_percentage'] else 1,
            help="Switch between viewing actual currency values or ROI percentages based on initial investment"
        )
        
        # Update the show_percentage flag based on the radio selection
        show_percentage = (display_mode == "📈 ROI Percentages (return on investment)")
        st.session_state['show_percentage'] = show_percentage
        
        # Allow users to input their initial investment amount
        if 'initial_investment' not in st.session_state:
            st.session_state['initial_investment'] = 10000.0  # Default value
        
        st.markdown("**💼 Initial Investment:**")    
        initial_investment = st.number_input(
            "Amount (AUD)",
            min_value=1.0, 
            value=st.session_state['initial_investment'],
            help="Your initial working capital in AUD. Used to calculate ROI percentages."
        )
        st.session_state['initial_investment'] = initial_investment
        
        if show_percentage:
            # Make sure we have the currency symbol before using it
            currency_symbol = currency_symbols.get(display_currency, display_currency)
            converted_investment = initial_investment * currency_rate
            st.info(f"Displaying profits as ROI percentages based on initial investment of {currency_symbol}{converted_investment:.2f}")

if uploaded_file is not None:
    # Process the data
    # Create a boolean column to check if CryptoValet is the buyer or seller
    filtered_df['is_buyer'] = filtered_df['Buyer'] == 'CryptoValet'
    
    # Calculate price per unit of cryptocurrency
    filtered_df['price_per_unit'] = filtered_df['Market Value'] / filtered_df['Crypto Amount']
    
    # Calculate trading fee in AUD
    filtered_df['fee_in_aud'] = filtered_df['Trading Fee'] * filtered_df['price_per_unit']
    
    # Calculate profit based on whether CryptoValet is buyer or seller, and subtract the fee
    filtered_df['profit'] = np.where(
        filtered_df['is_buyer'],
        filtered_df['Market Value'] - filtered_df['Local Currency Amount'] - filtered_df['fee_in_aud'],  # When buyer
        filtered_df['Local Currency Amount'] - filtered_df['Market Value'] - filtered_df['fee_in_aud']   # When seller
    )
    
    # Filter for completed trades to calculate core metrics
    # First let's analyze what status values we actually have in the data
    status_values = filtered_df['Status'].unique()
    st.sidebar.write("Debug - Available Status Values:", status_values)
    
    # Try to be flexible with the status matching - check for any status containing 'complet' (case insensitive)
    completed_df = filtered_df[filtered_df['Status'].str.lower().str.contains('complet')].copy()
    
    # If that didn't work, try the original logic
    if completed_df.empty and 'Completed' in status_values:
        completed_df = filtered_df[filtered_df['Status'] == 'Completed'].copy()
    
    # If still empty, show a warning
    if completed_df.empty:
        st.warning(f"No completed trades found in the selected period. Available statuses: {', '.join(status_values)}")
    
    st.sidebar.write(f"Debug - Found {len(completed_df)} completed trades")
    
    if not completed_df.empty:
        # Calculate total profit from completed trades
        total_profit = completed_df['profit'].sum()
    
        # Calculate trading days
        trading_days = completed_df['Time completed'].dt.date.nunique()
    
        # Calculate average daily profit
        avg_daily_profit = total_profit / trading_days if trading_days > 0 else 0
    
        # Calculate total completed trades
        total_completed_trades = len(completed_df)
    
        # Prepare data for daily profit analysis (used in Tab 1 and Prophet)
        daily_profit = completed_df.groupby(completed_df['Time completed'].dt.date).agg(Profit=('profit', 'sum')).reset_index()
        daily_profit.rename(columns={'Time completed': 'Date'}, inplace=True)
    
        # Prepare data for volume analysis (used in Tab 2)
        # We need to handle cases where there are only buys or only sells on a given day
        volume_agg = completed_df.groupby([completed_df['Time completed'].dt.date, 'is_buyer'])['Market Value'].sum().unstack(fill_value=0)
        volume_agg.columns = ['Sell Volume' if col is False else 'Buy Volume' for col in volume_agg.columns] # Rename columns based on boolean
        # Ensure both columns exist even if one type of trade is missing
        if 'Buy Volume' not in volume_agg:
            volume_agg['Buy Volume'] = 0
        if 'Sell Volume' not in volume_agg:
            volume_agg['Sell Volume'] = 0
        volume_data = volume_agg.reset_index().rename(columns={'Time completed': 'Date'})
    
    else:
        # Handle case with no completed trades in the filtered range
        total_profit = 0
        trading_days = 0
        avg_daily_profit = 0
        total_completed_trades = 0
        daily_profit = pd.DataFrame(columns=['Date', 'Profit']) # Empty dataframe with correct columns
        volume_data = pd.DataFrame(columns=['Date', 'Buy Volume', 'Sell Volume']) # Empty dataframe with correct columns

    # Ensure we're using the correct currency settings from session state
    currency_rate = st.session_state.get('currency_rate', 1.0)
    display_currency = st.session_state.get('display_currency', 'AUD')
    currency_symbol = st.session_state.get('currency_symbol', 'A$')
    
    # Add a subtle currency indicator
    if display_currency != 'AUD':
        st.info(f"📊 Viewing data in **{display_currency}** (Rate: 1 AUD = {currency_rate} {display_currency})")
    
    # Apply currency conversion to metrics
    converted_total_profit = total_profit * currency_rate
    converted_avg_daily_profit = avg_daily_profit * currency_rate
    converted_avg_profit_per_trade = (completed_df['profit'].mean() if not completed_df.empty else 0) * currency_rate
    
    # Calculate ROI percentages
    roi_total = (total_profit / initial_investment) * 100 if initial_investment > 0 else 0
    roi_daily_avg = (avg_daily_profit / initial_investment) * 100 if initial_investment > 0 else 0
    roi_per_trade = ((completed_df['profit'].mean() if not completed_df.empty else 0) / initial_investment) * 100 if initial_investment > 0 else 0
    
    # Define columns for the main metrics display
    col1, col2, col3, col4, col5 = st.columns(5)

    # Determine if showing currency or percentage based on toggle
    if show_percentage:
        with col1:
            st.metric("Total ROI", f"{roi_total:.2f}%")
        
        with col2:
            st.metric("Avg Daily ROI", f"{roi_daily_avg:.4f}%")
        
        with col3:
            st.metric("Trading Days", f"{trading_days}")
        
        with col4:
            st.metric("Completed Trades", f"{total_completed_trades}")
        
        with col5:
            st.metric("Avg ROI Per Trade", f"{roi_per_trade:.4f}%")
    else:
        with col1:
            st.metric("Total Profit", f"{currency_symbol}{converted_total_profit:.2f}")
        
        with col2:
            st.metric("Avg Daily Profit", f"{currency_symbol}{converted_avg_daily_profit:.2f}")
        
        with col3:
            st.metric("Trading Days", f"{trading_days}")
        
        with col4:
            st.metric("Completed Trades", f"{total_completed_trades}")
        
        with col5:
            st.metric("Avg Profit Per Trade", f"{currency_symbol}{converted_avg_profit_per_trade:.2f}")
    
    # Tabs for different charts and analyses
    tab1, tab2, tab3 = st.tabs(["Profit Analysis", "Volume Analysis", "Trade Details"])
    
    with tab1:
        st.header("Daily Profit Analysis")
        
        if not daily_profit.empty:
            # Create a copy of daily profit data with converted values
            converted_daily_profit = daily_profit.copy()
            
            if show_percentage:
                # Calculate ROI percentage for each day
                converted_daily_profit['Profit'] = (converted_daily_profit['Profit'] / initial_investment) * 100 if initial_investment > 0 else 0
                y_axis_label = 'ROI (%)'
                title_text = 'Daily ROI (%)'
            else:
                # Apply currency conversion - ensure we're using the correct rate from session state
                currency_rate = st.session_state.get('currency_rate', 1.0)
                display_currency = st.session_state.get('display_currency', 'AUD')
                
                converted_daily_profit['Profit'] = converted_daily_profit['Profit'] * currency_rate
                y_axis_label = f'Profit ({display_currency})'
                title_text = f'Daily Profit ({display_currency})'
            
            # Create an interactive bar chart for daily profit
            fig = px.bar(
                converted_daily_profit, 
                x='Date', 
                y='Profit',
                title=title_text,
                labels={'Profit': y_axis_label, 'Date': 'Date'},
                color='Profit',
                # Custom color scale: red for negative, different shades of green for positive
                color_continuous_scale=[
                    [0, 'rgb(255,0,0)'],     # Red for negative values
                    [0.5, 'rgb(255,0,0)'],    # Red at zero
                    [0.5, 'rgb(0,128,0)'],    # Green at zero
                    [1, 'rgb(0,255,0)']       # Brighter green for highest values
                ],
                color_continuous_midpoint=0,  # Set the midpoint at zero
                text_auto='.2f'  # Show profit value on each bar
            )
            
            # Update text size and position
            fig.update_traces(
                textfont=dict(size=14),  # Increase text size
                textposition='outside'    # Place text above bars
            )
            
            # Add a zero line to highlight profit/loss boundary
            fig.add_shape(
                type="line",
                x0=daily_profit['Date'].min(),
                y0=0,
                x1=daily_profit['Date'].max(),
                y1=0,
                line=dict(color="black", width=1.5, dash="dot")
            )
            
            # Format the chart
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                hoverlabel=dict(bgcolor="white", font_size=14),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Most profitable day
            if len(daily_profit) > 0:
                most_profitable_day = daily_profit.loc[daily_profit['Profit'].idxmax()]
                if show_percentage:
                    roi_most_profitable = (most_profitable_day['Profit'] / initial_investment) * 100 if initial_investment > 0 else 0
                    st.info(f"Most profitable day: {most_profitable_day['Date']} with {roi_most_profitable:.2f}% ROI")
                else:
                    converted_profit = most_profitable_day['Profit'] * currency_rate
                    st.info(f"Most profitable day: {most_profitable_day['Date']} with {currency_symbol}{converted_profit:.2f} profit")
            
            # Integrate Prophet forecast if available
            if prophet_available and len(daily_profit) >= 5:  # Need at least 5 data points for a reasonable forecast
                st.header("Profit Forecast")
                
                # Prepare data for Prophet
                forecast_data = daily_profit.rename(columns={'Date': 'ds', 'Profit': 'y'})
                forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
                
                # Set default forecast parameters
                forecast_days = min(30, max(14, len(forecast_data)))  # At least 14 days, at most 30, or match data length
                seasonality_mode = "additive"
                changepoint_prior_scale = 0.05
                
                with st.spinner("Generating profit forecast..."):
                    try:
                        # Create and fit the model
                        m = Prophet(
                            seasonality_mode=seasonality_mode,
                            changepoint_prior_scale=changepoint_prior_scale,
                            daily_seasonality=False,
                            weekly_seasonality=True if len(forecast_data) >= 14 else False,
                            yearly_seasonality=False
                        )
                        
                        # Add weekly seasonality if we have enough data
                        if len(forecast_data) >= 14:
                            m.add_seasonality(name='weekly', period=7, fourier_order=3)
                        
                        m.fit(forecast_data)
                        
                        # Create future dataframe
                        future = m.make_future_dataframe(periods=forecast_days)
                        
                        # Make predictions
                        forecast = m.predict(future)
                        
                        # Apply currency conversion to forecast
                        forecast_display = forecast.copy()
                        
                        # Ensure correct currency settings
                        currency_rate = st.session_state.get('currency_rate', 1.0)
                        display_currency = st.session_state.get('display_currency', 'AUD')
                        
                        for col in ['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']:
                            if col in forecast_display.columns:
                                if show_percentage:
                                    forecast_display[col] = (forecast_display[col] / initial_investment) * 100 if initial_investment > 0 else 0
                                else:
                                    forecast_display[col] = forecast_display[col] * currency_rate
                        
                        # Create a custom Plotly figure instead of using the built-in function
                        if show_percentage:
                            title_text = 'ROI Forecast (%)'
                            y_axis_label = 'ROI (%)'
                        else:
                            title_text = f'Profit Forecast ({display_currency})'
                            y_axis_label = f'Profit ({display_currency})'
                        
                        # Prepare the historical data for overlay
                        historical_data = forecast_data.copy()
                        # Apply conversion to historical data
                        if show_percentage:
                            historical_data['y'] = (historical_data['y'] / initial_investment) * 100 if initial_investment > 0 else 0
                        else:
                            historical_data['y'] = historical_data['y'] * currency_rate
                        
                        # Create a custom Plotly figure for enhanced visualization
                        fig = go.Figure()
                        
                        # Add confidence interval as a filled area
                        fig.add_trace(go.Scatter(
                            name='Upper Bound',
                            x=forecast_display['ds'],
                            y=forecast_display['yhat_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            name='Lower Bound',
                            x=forecast_display['ds'],
                            y=forecast_display['yhat_lower'],
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(216, 102, 67, 0.39)',  # Light yellow with transparency
                            fill='tonexty',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Add the forecast line
                        fig.add_trace(go.Scatter(
                            name='Forecast',
                            x=forecast_display['ds'],
                            y=forecast_display['yhat'],
                            mode='lines',
                            line=dict(color='yellow', width=2, dash='dot'),  # Changed to yellow for better contrast
                            hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>{y_axis_label}: %{{y:.2f}}<extra></extra>'
                        ))
                        
                        # Add the historical actual values as scatter points
                        fig.add_trace(go.Scatter(
                            name='Actual',
                            x=historical_data['ds'],
                            y=historical_data['y'],
                            mode='lines',
                            line=dict(color='cyan', width=2),  # Changed to cyan for better contrast
                            hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>Actual {y_axis_label}: %{{y:.2f}}<extra></extra>'
                        ))
                        
                        # Add changepoints if available
                        if 'delta' in forecast.columns and len(m.changepoints) > 0:
                            # Get the changepoint dates that are within our dataset
                            change_points = m.changepoints
                            
                            # Only include changepoints in our date range
                            change_points = change_points[(change_points >= historical_data['ds'].min()) & 
                                                        (change_points <= historical_data['ds'].max())]
                            
                            if len(change_points) > 0:
                                # Add vertical lines for changepoints
                                for cp in change_points:
                                    fig.add_shape(
                                        type="line",
                                        x0=cp,
                                        y0=forecast_display['yhat_lower'].min(),
                                        x1=cp,
                                        y1=forecast_display['yhat_upper'].max(),
                                        line=dict(color="gray", width=1, dash="dot"),
                                    )
                        
                        # Update layout with better formatting
                        fig.update_layout(
                            title=title_text,
                            xaxis_title='Date',
                            yaxis_title=y_axis_label,
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5
                            ),
                            margin=dict(l=40, r=40, t=40, b=40),
                            hovermode="x unified"
                        )
                        
                        # Show the interactive plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add forecast components visualization - just weekly seasonality and trend side by side
                        st.subheader("Forecast Components")
                        
                        # Create a 1x2 subplot for the two components
                        components_fig = make_subplots(rows=1, cols=2, 
                                                    subplot_titles=("Weekly Pattern", "Overall Trend"),
                                                    horizontal_spacing=0.1)
                        
                        # Extract the weekly seasonality component
                        if 'weekly' in forecast_display.columns:
                            week_df = forecast_display[['ds', 'weekly']].copy()
                            # Group by day of week and take mean to simplify
                            week_df['day_of_week'] = week_df['ds'].dt.day_name()
                            # Get a consistent order of days
                            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            # Aggregate by day of week
                            weekly_pattern = week_df.groupby('day_of_week')['weekly'].mean().reindex(day_order).reset_index()
                            
                            # Add weekly component to first subplot
                            components_fig.add_trace(
                                go.Bar(
                                    x=weekly_pattern['day_of_week'],
                                    y=weekly_pattern['weekly'],
                                    marker_color='rgba(55, 83, 109, 0.7)',
                                    name='Day of Week Effect',
                                    hovertemplate='%{x}: %{y:.2f}<extra></extra>'
                                ),
                                row=1, col=1
                            )
                            
                            # Add zero reference line
                            components_fig.add_shape(
                                type='line',
                                x0=0, x1=1,
                                y0=0, y1=0,
                                xref='paper', yref='y1',
                                line=dict(color='black', width=1, dash='dot'),
                                row=1, col=1
                            )
                        
                        # Add trend component to second subplot
                        components_fig.add_trace(
                            go.Scatter(
                                x=forecast_display['ds'],
                                y=forecast_display['trend'],
                                mode='lines',
                                name='Trend',
                                line=dict(color='red', width=2),
                                hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>'
                            ),
                            row=1, col=2
                        )
                        
                        # Add trend confidence intervals if available
                        if 'trend_lower' in forecast_display.columns and 'trend_upper' in forecast_display.columns:
                            components_fig.add_trace(
                                go.Scatter(
                                    x=forecast_display['ds'],
                                    y=forecast_display['trend_upper'],
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ),
                                row=1, col=2
                            )
                            
                            components_fig.add_trace(
                                go.Scatter(
                                    x=forecast_display['ds'],
                                    y=forecast_display['trend_lower'],
                                    mode='lines',
                                    line=dict(width=0),
                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                    fill='tonexty',
                                    showlegend=False,
                                    hoverinfo='skip'
                                ),
                                row=1, col=2
                            )
                        
                        # Update layout for components
                        components_fig.update_layout(
                            height=350,
                            showlegend=False,
                            margin=dict(l=40, r=40, t=50, b=40)
                        )
                        
                        # Update y-axis titles
                        y_effect_label = 'ROI Effect (%)' if show_percentage else f'Profit Effect ({display_currency})'
                        components_fig.update_yaxes(title_text=y_effect_label, row=1, col=1)
                        components_fig.update_yaxes(title_text=y_axis_label, row=1, col=2)
                        
                        # X-axis formatting for trend
                        components_fig.update_xaxes(title_text='', row=1, col=2)
                        components_fig.update_xaxes(title_text='', row=1, col=1)
                        
                        # Show components
                        st.plotly_chart(components_fig, use_container_width=True)
                        
                        # Calculate forecast metrics
                        forecast_period = forecast[forecast['ds'] > forecast_data['ds'].max()]
                        
                        # Apply currency conversion to forecast metrics
                        converted_next_30_days = forecast_period[forecast_period['ds'] <= forecast_data['ds'].max() + pd.Timedelta(days=30)]
                        next_30_days_profit = converted_next_30_days['yhat'].sum() if not converted_next_30_days.empty else 0
                        avg_daily_forecast = forecast_period['yhat'].mean()
                        
                        # Prepare display values
                        if show_percentage:
                            # Calculate ROI percentages
                            display_30_days = f"{(next_30_days_profit / initial_investment) * 100:.2f}%" if initial_investment > 0 else "0.00%"
                            
                            projected_monthly = avg_daily_forecast * 30.5
                            display_monthly = f"{(projected_monthly / initial_investment) * 100:.2f}%" if initial_investment > 0 else "0.00%"
                            
                            annual_projection = avg_daily_forecast * 365
                            display_annual = f"{(annual_projection / initial_investment) * 100:.2f}%" if initial_investment > 0 else "0.00%"
                            
                            metric_labels = ["30-Day ROI Forecast", "Monthly ROI Projection", "Annual ROI Projection"]
                        else:
                            # Apply currency conversion
                            converted_next_30_days_profit = next_30_days_profit * currency_rate
                            
                            projected_monthly = avg_daily_forecast * 30.5
                            converted_projected_monthly = projected_monthly * currency_rate
                            
                            annual_projection = avg_daily_forecast * 365
                            converted_annual_projection = annual_projection * currency_rate
                            
                            display_30_days = f"{currency_symbol}{converted_next_30_days_profit:.2f}"
                            display_monthly = f"{currency_symbol}{converted_projected_monthly:.2f}"
                            display_annual = f"{currency_symbol}{converted_annual_projection:.2f}"
                            
                            metric_labels = ["30-Day Forecast", "Monthly Projection", "Annual Projection"]
                        
                        # Projections based on Prophet forecast
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(metric_labels[0], display_30_days)
                        
                        with col2:
                            st.metric(metric_labels[1], display_monthly)
                        
                        with col3:
                            st.metric(metric_labels[2], display_annual)
                    
                    except Exception as e:
                        st.error(f"Error in forecast: {str(e)}")
                        st.info("Unable to generate forecast. Try adjusting the date range to include more data.")
            
        else:
            st.warning("No completed trades in the selected date range to analyze profit.")
    
    with tab2:
        st.header("Trade Volume Analysis")
        
        if not volume_data.empty:
            # Get current currency settings
            currency_rate = st.session_state.get('currency_rate', 1.0)
            display_currency = st.session_state.get('display_currency', 'AUD')
            currency_symbol = st.session_state.get('currency_symbol', 'A$')
            
            # Create a copy with converted values
            converted_volume_data = volume_data.copy()
            converted_volume_data['Buy Volume'] = converted_volume_data['Buy Volume'] * currency_rate
            converted_volume_data['Sell Volume'] = converted_volume_data['Sell Volume'] * currency_rate
            
            # Create a stacked bar chart for volume data
            fig = go.Figure()
            
            # Add buy volume
            fig.add_trace(go.Bar(
                x=converted_volume_data['Date'],
                y=converted_volume_data['Buy Volume'],
                name='Buy Volume',
                marker_color='#4169E1'  # Royal Blue
            ))
            
            # Add sell volume
            fig.add_trace(go.Bar(
                x=converted_volume_data['Date'],
                y=converted_volume_data['Sell Volume'],
                name='Sell Volume',
                marker_color='#FF7F50'  # Coral
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Daily Trading Volume ({display_currency})',
                xaxis_title='Date',
                yaxis_title=f'Volume ({display_currency})',
                barmode='stack',  # Stack the bars instead of grouping
                xaxis_tickangle=-45,
                height=500,
                hoverlabel=dict(bgcolor="white", font_size=14),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add hover template to show values in currency format
            fig.update_traces(
                hovertemplate=f"Date: %{{x}}<br>Volume: {currency_symbol}%{{y:.2f}}<extra></extra>"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Buy vs Sell distribution
            if total_completed_trades > 0:
                buy_count = completed_df['is_buyer'].sum()
                sell_count = total_completed_trades - buy_count
                
                st.subheader("Buy vs Sell Distribution")
                
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    fig_pie = px.pie(
                        values=[buy_count, sell_count],
                        names=['Buy', 'Sell'],
                        title='Buy vs Sell Trades',
                        color_discrete_sequence=['#4169E1', '#FF7F50'],  # Royal Blue, Coral
                        hole=0.4
                    )
                    
                    fig_pie.update_traces(
                        textinfo='percent+value',
                        texttemplate='%{percent:.1%}<br>(%{value})'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Replace profit chart with volume by trade type
                    buy_volume = completed_df[completed_df['is_buyer']]['Market Value'].sum()
                    sell_volume = completed_df[~completed_df['is_buyer']]['Market Value'].sum()
                    
                    # Apply currency conversion
                    converted_buy_volume = buy_volume * currency_rate
                    converted_sell_volume = sell_volume * currency_rate
                    
                    # Calculate average volume per trade, handling potential division by zero
                    avg_buy_volume_per_trade = converted_buy_volume / buy_count if buy_count > 0 else 0
                    avg_sell_volume_per_trade = converted_sell_volume / sell_count if sell_count > 0 else 0

                    volume_data_summary = pd.DataFrame({
                        'Type': ['Buy', 'Sell'],
                        'Volume': [converted_buy_volume, converted_sell_volume],
                        'Count': [buy_count, sell_count],
                        'Avg Volume Per Trade': [avg_buy_volume_per_trade, avg_sell_volume_per_trade]
                    })
                    
                    fig_volume = px.bar(
                        volume_data_summary,
                        x='Type',
                        y='Volume',
                        color='Type',
                        color_discrete_sequence=['#4169E1', '#FF7F50'],  # Royal Blue, Coral
                        text_auto='.2f',
                        title=f'Total Volume by Trade Type ({display_currency})',
                        hover_data=['Count', 'Avg Volume Per Trade'] # Add hover data
                    )

                    # Customize hover template
                    fig_volume.update_traces(hovertemplate=(
                        f"<b>%{{x}}</b><br>"
                        f"Total Volume: {currency_symbol}%{{y:.2f}}<br>"
                        f"Trade Count: %{{customdata[0]}}<br>"
                        f"Avg Volume/Trade: {currency_symbol}%{{customdata[1]:.2f}}"
                        f"<extra></extra>"
                    ))
                    
                    fig_volume.update_layout(
                        yaxis_title=f'Volume ({display_currency})',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                
                # Add trading time heatmap
                st.subheader("Trading Time Heatmap")
                
                # Extract hour and day of week from completed trades
                completed_df['Hour'] = completed_df['Time completed'].dt.hour
                completed_df['Day of Week'] = completed_df['Time completed'].dt.day_name()
                
                # Create pivot table for heatmap
                heatmap_data = pd.pivot_table(
                    completed_df,
                    values='Market Value',
                    index='Hour',
                    columns='Day of Week',
                    aggfunc='count',
                    fill_value=0
                )
                
                # Reorder days of week
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = heatmap_data.reindex(columns=days_order)
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x='Day of Week', y='Hour of Day', color='Number of Trades'),
                    aspect='auto',
                    color_continuous_scale='YlOrRd'
                )
                
                fig_heatmap.update_layout(
                    title='Trading Activity by Hour and Day',
                    xaxis_title='Day of Week',
                    yaxis_title='Hour of Day (24h)',
                    height=500
                )
                
                # Update y-axis to show all hours
                fig_heatmap.update_yaxes(tickmode='linear', tick0=0, dtick=1)
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No trade volume data available for the selected date range.")
    
    with tab3:
        st.header("Trade Details")
        
        # Add filters for more detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            cryptocurrencies = filtered_df['Cryptocurrency'].unique()
            currency_filter = st.multiselect("Filter by Cryptocurrency", cryptocurrencies, default=cryptocurrencies)
        
        with col2:
            local_currencies = filtered_df['Local Currency'].unique()
            currency_type_filter = st.multiselect("Filter by Local Currency", local_currencies, default=local_currencies)
        
        # Apply filters
        detail_df = filtered_df
        if currency_filter:
            detail_df = detail_df[detail_df['Cryptocurrency'].isin(currency_filter)]
        if currency_type_filter:
            detail_df = detail_df[detail_df['Local Currency'].isin(currency_type_filter)]
        
        # Show the detailed data
        detail_df_display = detail_df.copy()
        
        # Get current currency settings
        currency_rate = st.session_state.get('currency_rate', 1.0)
        display_currency = st.session_state.get('display_currency', 'AUD')
        currency_symbol = st.session_state.get('currency_symbol', 'A$')
        
        # Convert columns that need currency conversion
        for column in ['Market Value', 'Local Currency Amount', 'fee_in_aud', 'profit']:
            if column in detail_df_display.columns:
                detail_df_display[column] = detail_df_display[column] * currency_rate
        
        # Rename fee column to match display currency
        if 'fee_in_aud' in detail_df_display.columns:
            detail_df_display = detail_df_display.rename(columns={'fee_in_aud': f'fee_in_{display_currency.lower()}'})
        
        # Choose columns to display based on percentage toggle
        if show_percentage:
            display_columns = [
                'Time completed', 'Status', 'Cryptocurrency', 'Crypto Amount',
                'Market Value', 'Local Currency Amount', 'Local Currency',
                'roi_percentage', 'Buyer', 'Seller'
            ]
            # Format the ROI percentage for display
            detail_df_display['roi_percentage'] = detail_df_display['roi_percentage'].apply(lambda x: f"{x:.4f}%")
        else:
            display_columns = [
                'Time completed', 'Status', 'Cryptocurrency', 'Crypto Amount', 
                'Market Value', 'Local Currency Amount', 'Local Currency',
                f'fee_in_{display_currency.lower()}', 'profit', 'Buyer', 'Seller'
            ]
        
        # Add ROI column if showing percentages, ensuring it exists first
        if 'profit' in detail_df_display.columns and initial_investment > 0:
            detail_df_display['roi_percentage'] = (detail_df_display['profit'] / (initial_investment * currency_rate)) * 100 # Use converted investment for ROI calc
        else:
            detail_df_display['roi_percentage'] = 0.0 # Or handle as NaN/None

        # Ensure all columns in display_columns actually exist in detail_df_display before trying to display them
        final_display_columns = [col for col in display_columns if col in detail_df_display.columns]

        st.dataframe(
            detail_df_display[final_display_columns].sort_values('Time completed', ascending=False),
            hide_index=True,
            use_container_width=True
        )
        
        # Add a download button for the filtered data
        st.download_button(
            label=f"Download Filtered Data as CSV ({display_currency})",
            data=detail_df_display.to_csv(index=False).encode('utf-8'),
            file_name=f'filtered_p2p_data_{display_currency}.csv',
            mime='text/csv',
        )
else:
    st.info("👈 Please upload your CSV file to start the analysis")
    
    # Example data placeholder
    st.markdown("""
    ### Expected CSV Format
    
    Your CSV should include the following columns:
    - Time completed
    - Time created
    - Status
    - Buyer
    - Seller
    - Currency
    - Crypto Amount
    - Market Value
    - Local Currency Amount
    - Trading Fee
    - Payment method
    
    Upload a file to begin your analysis.
    """) 
