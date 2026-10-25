import asyncio
import traceback
from app.core.config import get_settings
from app.db.session import _get_session_factory
from app.services.job_service import create_job

async def main():
    try:
        factory = _get_session_factory()
        async with factory() as session:
            job = await create_job(
                session,
                input_type="script",
                script="Autonomous AI transforms video generation.",
                aspect_ratios=["16:9"]
            )
            print("JOB SUCCESS ID:", job.id)
    except Exception as e:
        traceback.print_exc()

asyncio.run(main())
