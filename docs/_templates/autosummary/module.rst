{# Copyright (C) 2026 RidgeRun, LLC (http://www.ridgerun.com)
   SPDX-License-Identifier: AGPL-3.0-only #}

{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

{% if modules %}
.. autosummary::
   :toctree:
   :recursive:

{% for module in modules %}
   {{ module }}
{% endfor %}
{% endif %}
