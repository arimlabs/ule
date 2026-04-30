# Auto-discover and import all model modules
import importlib
import pkgutil
from pathlib import Path

# Auto-import all modules in this package
package_dir = Path(__file__).parent
for (_, module_name, _) in pkgutil.iter_modules([str(package_dir)]):
    importlib.import_module(f"{__name__}.{module_name}")