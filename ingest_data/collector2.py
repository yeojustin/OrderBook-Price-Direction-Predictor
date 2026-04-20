import asyncio
import websockets
import json
import csv
import time
import os

SYMBOL = "ethusdt"
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth5@1000ms"
CSV_FILE = f"data/l2_data_{SYMBOL.upper()}.csv"

def setup_csv():
    headers = ['timestamp', 'mid_price'] + \
              [f'b_p_{i}' for i in range(5)] + [f'b_q_{i}' for i in range(5)] + \
              [f'a_p_{i}' for i in range(5)] + [f'a_q_{i}' for i in range(5)]
    
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

async def stream_data():
    setup_csv()
    print(f"Connecting to {STREAM_URL}...")
    
    async with websockets.connect(STREAM_URL) as ws:
        print("Connected. Logging data...")
        
        with open(CSV_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            while True:
                try:
                    response = await ws.recv()
                    data = json.loads(response)
                    
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])
                    
                    if not bids or not asks:
                        continue
                        
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_bid + best_ask) / 2
                    
                    row = [time.time(), round(mid_price, 2)]
                    row += [float(p) for p, q in bids] + [float(q) for p, q in bids]
                    row += [float(p) for p, q in asks] + [float(q) for p, q in asks]
                    
                    writer.writerow(row)
                    f.flush() 
                    
                    print(f"[{time.strftime('%H:%M:%S')}] {SYMBOL.upper()} Mid: {mid_price:.2f}")
                    
                except Exception as e:
                    print(f"Error: {e}")
                    await asyncio.sleep(1) 

if __name__ == "__main__":
    try:
        asyncio.run(stream_data())
    except KeyboardInterrupt:
        print("\nStopped.")