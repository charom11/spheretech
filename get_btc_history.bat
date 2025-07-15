@echo off
REM Batch file to set Binance API credentials and run get_btc_history.py

REM === EDIT THESE LINES WITH YOUR CREDENTIALS ===
set BINANCE_API_KEY=4JKNtCnmBkY9qmcPFnPIG9hy8TRYfr0CdrejKLvKWIxrKUbiqDNprqLrmxx6sa98
set BINANCE_API_SECRET=k9mZg6fiXTndY2ijS6GDutvcFCyoA326RvCHVXc34bHimq57OxweUJAKbFiJdOOr
REM =============================================

REM Run the Python script
python get_btc_history.py

pause 