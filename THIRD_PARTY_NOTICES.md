# Third-party dependencies

The Apache 2.0 grant for this project does not relicense third-party packages.
The following runtime dependencies are installed separately:

| Dependency | Role | Upstream |
| --- | --- | --- |
| NumPy | Arrays and numerical arithmetic | https://numpy.org/ |
| SciPy | DOP853 integrators, optimization and numerical support | https://scipy.org/ |
| Matplotlib | Optional figures | https://matplotlib.org/ |

Their distributions contain their own license and attribution information.
If distributing those packages or their binaries, preserve their applicable
licenses and notices. They are not bundled into this release.

`native/integrator/_extended_dop853.py` is the project's precision wrapper
around the separately installed SciPy DOP853 class. It imports SciPy rather
than including SciPy's solver source or claiming a new Runge-Kutta method.

`LICENSE` reproduces the standard Apache License 2.0 text. Its authoritative
reference is https://www.apache.org/licenses/LICENSE-2.0.txt.
