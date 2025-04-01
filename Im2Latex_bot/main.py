import asyncio
from tg_client import TGClient

if __name__ == '__main__':
    # executor.start_polling(dp, skip_updates=True)
    client = TGClient()
    asyncio.run(client.start())
