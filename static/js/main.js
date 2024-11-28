// Global variables for storing performance data
let performanceData = null;

// Function to update the live signal display
function updateLiveSignal() {
    $.get('/api/current_signal', function(data) {
        const marketStatus = $('#market-status');
        if (data.is_market_open) {
            marketStatus.html('Market is Open').removeClass('closed').addClass('open');
        } else {
            marketStatus.html('Market is Closed').removeClass('open').addClass('closed');
        }

        if (data.signal !== null) {
            let signalText = 'HOLD';
            if (data.signal === 1) signalText = 'BUY (CALL)';
            if (data.signal === -1) signalText = 'SELL (PUT)';
            
            $('#current-signal').text(signalText);
            $('#current-price').text(data.price ? data.price.toFixed(2) : '-');
            $('#trailing-stop').text(data.trailing_stop ? data.trailing_stop.toFixed(2) : '-');
            $('#last-updated').text(data.timestamp || '-');
            
            // Update PCR information
            if (data.pcr_data) {
                const pcrData = data.pcr_data;
                $('#current-pcr').text(pcrData.current_pcr.toFixed(2));
                
                const pcrChange = pcrData.pcr_change;
                const pcrChangeElement = $('#pcr-change');
                pcrChangeElement.text(pcrChange.toFixed(2));
                pcrChangeElement.removeClass('positive-return negative-return');
                if (pcrChange > 0) pcrChangeElement.addClass('positive-return');
                if (pcrChange < 0) pcrChangeElement.addClass('negative-return');
                
                $('#pcr-ma5').text(pcrData.pcr_ma5.toFixed(2));
                $('#pcr-ma20').text(pcrData.pcr_ma20 ? pcrData.pcr_ma20.toFixed(2) : '-');
            }
        }
    });
}

// Function to update the performance plot
function updatePerformancePlot() {
    $.get('/api/historical_performance', function(data) {
        performanceData = data;
        
        // Create cumulative returns trace
        const cumulativeReturns = data.returns.reduce((acc, val) => {
            const prev = acc.length > 0 ? acc[acc.length - 1] : 0;
            acc.push(prev + (val || 0));
            return acc;
        }, []);

        const traces = [
            {
                x: data.dates,
                y: data.prices,
                name: 'Nifty 50',
                type: 'scatter',
                line: { color: '#1f77b4' }
            },
            {
                x: data.dates,
                y: data.trailing_stops,
                name: 'Trailing Stop',
                type: 'scatter',
                line: { color: '#ff7f0e', dash: 'dot' }
            },
            {
                x: data.dates,
                y: cumulativeReturns,
                name: 'Cumulative Returns (%)',
                type: 'scatter',
                yaxis: 'y2',
                line: { color: '#2ca02c' }
            }
        ];

        const layout = {
            title: 'Trading Performance',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Price', side: 'left' },
            yaxis2: {
                title: 'Cumulative Returns (%)',
                overlaying: 'y',
                side: 'right'
            },
            showlegend: true,
            legend: { x: 0, y: 1 }
        };

        Plotly.newPlot('performance-plot', traces, layout);
        updateTradesTable();
    });
}

// Function to update the trades table
function updateTradesTable() {
    if (!performanceData) return;

    const tbody = $('#trades-body');
    tbody.empty();

    let currentPosition = null;
    let entryPrice = null;
    let entryDate = null;

    for (let i = 0; i < performanceData.signals.length; i++) {
        const signal = performanceData.signals[i];
        const price = performanceData.prices[i];
        const date = performanceData.dates[i];

        if (signal !== 0 && currentPosition === null) {
            // Opening a position
            currentPosition = signal;
            entryPrice = price;
            entryDate = date;
        } else if (currentPosition !== null && (signal === 0 || signal === -currentPosition)) {
            // Closing a position
            const returns = ((price - entryPrice) / entryPrice * 100 * currentPosition).toFixed(2);
            const returnClass = returns >= 0 ? 'positive-return' : 'negative-return';

            tbody.append(`
                <tr>
                    <td>${entryDate}</td>
                    <td>${currentPosition === 1 ? 'BUY (CALL)' : 'SELL (PUT)'}</td>
                    <td>${entryPrice.toFixed(2)}</td>
                    <td>${price.toFixed(2)}</td>
                    <td class="${returnClass}">${returns}%</td>
                </tr>
            `);

            currentPosition = null;
            entryPrice = null;
            entryDate = null;
        }
    }
}

// Initialize the dashboard
$(document).ready(function() {
    // Initial updates
    updateLiveSignal();
    updatePerformancePlot();

    // Set up periodic updates
    setInterval(updateLiveSignal, 5000);  // Update live signal every 5 seconds
    setInterval(updatePerformancePlot, 300000);  // Update performance data every 5 minutes
});
