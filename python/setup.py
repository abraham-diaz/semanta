from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Tags the wheel as platform-specific: it bundles a precompiled native
    SQLite extension, not pure Python code, so it can't be tagged `py3-none-any`."""

    def has_ext_modules(self):
        return True


setup(
    name="semanta",
    version="0.1.0",
    description="Python loader for the Semanta SQLite extension.",
    license="MIT",
    packages=["semanta"],
    package_data={"semanta": ["*.so", "*.dylib", "*.dll"]},
    include_package_data=True,
    python_requires=">=3.8",
    distclass=BinaryDistribution,
)
