# Refactored architecture
from src.real_time.frontend_opencv import OpenCVFrontendController
from src.real_time.frontend_matplotlib import MatplotlibFrontendController
# Old architecture
from src.real_time.base_backend import BackendInterface
from src.real_time.background_backend import BackgroundLoop
from src.real_time.matplotlib_backend import MatplotlibLoop
from src.real_time.text_input_backend import TextInputLoop
