import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

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
    st.header("Upload Data")
    
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
        st.header("Date Filter")
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
    
    # Filter to only completed trades for profit analysis
    completed_df = filtered_df[filtered_df['Status'] == 'COMPLETED'].copy()
    
    # Group by date and calculate daily profit
    daily_profit = completed_df.groupby(completed_df['Time completed'].dt.date)['profit'].sum().reset_index()
    daily_profit.columns = ['Date', 'Profit']
    
    # Prepare data for the volume chart
    def prepare_volume_data():
        # Group all trades by date
        all_trades = filtered_df.copy()
        all_trades['Date'] = all_trades['Time completed'].dt.date
        
        # Calculate buy and sell volumes separately
        buy_volume = all_trades[all_trades['is_buyer']].groupby('Date')['Market Value'].sum()
        sell_volume = all_trades[~all_trades['is_buyer']].groupby('Date')['Market Value'].sum()
        
        # Combine into a single DataFrame
        volume_data = pd.DataFrame({
            'Buy Volume': buy_volume,
            'Sell Volume': sell_volume
        }).fillna(0).reset_index()
        
        return volume_data

    volume_data = prepare_volume_data()
    
    # Calculate overall statistics
    total_profit = completed_df['profit'].sum()
    avg_daily_profit = daily_profit['Profit'].mean() if not daily_profit.empty else 0
    total_completed_trades = len(completed_df)
    trading_days = len(daily_profit) if not daily_profit.empty else 0
    
    # Main content area
    # Key metrics at the top
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Profit", f"${total_profit:.2f}")
    
    with col2:
        st.metric("Avg Daily Profit", f"${avg_daily_profit:.2f}")
    
    with col3:
        st.metric("Trading Days", f"{trading_days}")
    
    with col4:
        st.metric("Completed Trades", f"{total_completed_trades}")
    
    with col5:
        avg_profit_per_trade = completed_df['profit'].mean() if not completed_df.empty else 0
        st.metric("Avg Profit Per Trade", f"${avg_profit_per_trade:.2f}")
    
    # Add Prophet-based projections if available
    if prophet_available and len(daily_profit) >= 5:  # Need at least 5 data points for a reasonable forecast
        st.markdown("### Projected Profit")
        
        # Prepare data for Prophet
        forecast_data = daily_profit.rename(columns={'Date': 'ds', 'Profit': 'y'})
        forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
        
        # Set default forecast parameters
        forecast_days = min(30, max(14, len(forecast_data)))
        
        try:
            # Create and fit the model
            m = Prophet(
                seasonality_mode="additive",
                changepoint_prior_scale=0.05,
                daily_seasonality=False,
                weekly_seasonality=True if len(forecast_data) >= 14 else False,
                yearly_seasonality=False
            )
            
            if len(forecast_data) >= 14:
                m.add_seasonality(name='weekly', period=7, fourier_order=3)
            
            m.fit(forecast_data)
            future = m.make_future_dataframe(periods=forecast_days)
            forecast = m.predict(future)
            
            # Calculate projections
            forecast_period = forecast[forecast['ds'] > forecast_data['ds'].max()]
            next_30_days = forecast_period[forecast_period['ds'] <= forecast_data['ds'].max() + pd.Timedelta(days=30)]
            next_30_days_profit = next_30_days['yhat'].sum() if not next_30_days.empty else 0
            avg_daily_forecast = forecast_period['yhat'].mean()
            
            # Display projections
            col1, col2, col3 = st.columns([2, 2, 4])  # Use weighted columns to push metrics together
            
            with col1:
                st.metric("30-Day Forecast", f"${next_30_days_profit:.2f}",
                         help="Direct forecast of profit for the next 30 days based on time series analysis")
            
            with col2:
                annual_projection = next_30_days_profit * (365/30)  # Scale up from 30-day forecast
                st.metric("Annual Projection", f"${annual_projection:.2f}",
                         help="Annualized projection based on 30-day forecast (30-day forecast × 12.17)")
            
            with col3:
                # Empty column to push metrics together
                pass
        
        except Exception as e:
            st.error("Unable to generate projections. Try adjusting the date range to include more data.")
    else:
        st.markdown("### Projected Profit")
        st.info("Insufficient data for generating projections. Please select a date range with at least 5 days of trading data.")
    
    # Tabs for different charts and analyses
    tab1, tab2, tab3 = st.tabs(["Profit Analysis", "Volume Analysis", "Trade Details"])
    
    with tab1:
        st.header("Daily Profit Analysis")
        
        if not daily_profit.empty:
            # Create an interactive bar chart for daily profit
            fig = px.bar(
                daily_profit, 
                x='Date', 
                y='Profit',
                title='Daily Profit',
                labels={'Profit': 'Profit (AUD)', 'Date': 'Date'},
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
                st.info(f"Most profitable day: {most_profitable_day['Date']} with ${most_profitable_day['Profit']:.2f} profit")
            
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
                        
                        # Create Plotly figure for interactive visualization
                        fig = plot_plotly(m, forecast, trend=True, changepoints=True)
                        
                        # Update layout
                        fig.update_layout(
                            title='Profit Forecast',
                            xaxis_title='Date',
                            yaxis_title='Profit (AUD)',
                            height=500,
                            legend_title_text='Legend'
                        )
                        
                        # Show the interactive plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add forecast components visualization
                        st.subheader("Forecast Components")
                        
                        # Create components plot
                        fig_comp = plot_components_plotly(m, forecast)
                        
                        # Update layout
                        fig_comp.update_layout(
                            height=400 * len(fig_comp.data),  # Adjust height based on number of components
                            showlegend=False
                        )
                        
                        # Show the components plot
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Calculate forecast metrics
                        forecast_period = forecast[forecast['ds'] > forecast_data['ds'].max()]
                        
                        # Projections based on Prophet forecast
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Calculate monthly projection (next 30 days)
                            next_30_days = forecast_period[forecast_period['ds'] <= forecast_data['ds'].max() + pd.Timedelta(days=30)]
                            next_30_days_profit = next_30_days['yhat'].sum() if not next_30_days.empty else 0
                            st.metric("30-Day Forecast", f"${next_30_days_profit:.2f}")
                        
                        with col2:
                            avg_daily_forecast = forecast_period['yhat'].mean()
                            projected_monthly = avg_daily_forecast * 30.5
                            st.metric("Monthly Projection", f"${projected_monthly:.2f}")
                        
                        with col3:
                            annual_projection = avg_daily_forecast * 365
                            st.metric("Annual Projection", f"${annual_projection:.2f}")
                    
                    except Exception as e:
                        st.error(f"Error in forecast: {str(e)}")
                        st.info("Unable to generate forecast. Try adjusting the date range to include more data.")
            
        else:
            st.warning("No completed trades in the selected date range to analyze profit.")
    
    with tab2:
        st.header("Trade Volume Analysis")
        
        if not volume_data.empty:
            # Create a stacked bar chart for volume data
            fig = go.Figure()
            
            # Add buy volume
            fig.add_trace(go.Bar(
                x=volume_data['Date'],
                y=volume_data['Buy Volume'],
                name='Buy Volume',
                marker_color='#4169E1'  # Royal Blue
            ))
            
            # Add sell volume
            fig.add_trace(go.Bar(
                x=volume_data['Date'],
                y=volume_data['Sell Volume'],
                name='Sell Volume',
                marker_color='#FF7F50'  # Coral
            ))
            
            # Update layout
            fig.update_layout(
                title='Daily Trading Volume (AUD)',
                xaxis_title='Date',
                yaxis_title='Volume (AUD)',
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
                hovertemplate="Date: %{x}<br>Volume: $%{y:.2f}<extra></extra>"
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
                    
                    volume_data = pd.DataFrame({
                        'Type': ['Buy', 'Sell'],
                        'Volume': [buy_volume, sell_volume],
                        'Count': [buy_count, sell_count],
                        'Avg Volume': [
                            buy_volume / buy_count if buy_count > 0 else 0,
                            sell_volume / sell_count if sell_count > 0 else 0
                        ]
                    })
                    
                    fig_volume = px.bar(
                        volume_data,
                        x='Type',
                        y='Volume',
                        color='Type',
                        color_discrete_sequence=['#4169E1', '#FF7F50'],  # Royal Blue, Coral
                        text_auto='.2f',
                        title='Total Volume by Trade Type'
                    )
                    
                    fig_volume.update_layout(
                        yaxis_title='Volume (AUD)',
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
        st.dataframe(
            detail_df[[
                'Time completed', 'Status', 'Cryptocurrency', 'Crypto Amount', 
                'Market Value', 'Local Currency Amount', 'Local Currency',
                'fee_in_aud', 'profit', 'Buyer', 'Seller'
            ]].sort_values('Time completed', ascending=False),
            hide_index=True,
            use_container_width=True
        )
        
        # Add a download button for the filtered data
        st.download_button(
            label="Download Filtered Data as CSV",
            data=detail_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_p2p_data.csv',
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