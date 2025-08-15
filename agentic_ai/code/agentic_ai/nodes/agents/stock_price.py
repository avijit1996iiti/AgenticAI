import yfinance as yf

from agentic_ai.utils.agent_state import AgentState


def stock_price_tool(state: AgentState):
    print("-> Stock Price ->")
    query = state["messages"][0].lower().strip()

    # Basic mapping from company name to ticker
    company_to_ticker = {
        "nvidia": "NVDA",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "tesla": "TSLA",
        # Add more as needed
    }

    ticker = None
    for name in company_to_ticker:
        if name in query:
            ticker = company_to_ticker[name]
            break

    if not ticker:
        # fallback: assume user entered a ticker
        ticker = query.strip().upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get("longName", ticker)
        current_price = info.get("currentPrice")
        currency = info.get("financialCurrency", "USD")
        previous_close = info.get("previousClose")

        if current_price and previous_close:
            delta = round(current_price - previous_close, 2)
            change_percent = round((delta / previous_close) * 100, 2)
            direction = "up" if delta > 0 else "down" if delta < 0 else "unchanged"
            response = (
                f"The current stock price of {name} ({ticker}) is {current_price} {currency}, "
                f"{direction} {abs(delta)} {currency} ({abs(change_percent)}%) from the previous close."
            )
        elif current_price:
            response = f"The current stock price of {name} ({ticker}) is {current_price} {currency}."
        else:
            response = f"Could not fetch the current stock price for '{ticker}'. Please check the ticker symbol."

    except Exception as e:
        response = f"Error retrieving stock information for '{ticker}': {str(e)}"

    return {"messages": [response]}
