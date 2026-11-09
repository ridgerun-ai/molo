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
