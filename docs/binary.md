---
layout: default
title: Binary Classification Benchmarks
---

# Binary Classification Benchmarks

<style>
.wrapper {
  max-width: 1200px;
  padding-left: 2rem;
  padding-right: 2rem;
}

@media screen and (max-width: 640px) {
  .wrapper {
    padding-left: 1.25rem;
    padding-right: 1.25rem;
  }
}
</style>

{% assign datasets = site.data.datasets.binary %}

<h2>Dataset Overview</h2>

<div class="table-scroll">
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Instances</th>
      <th>Features</th>
      <th>Numeric</th>
      <th>Categorical</th>
      <th>Target</th>
      <th>Classes</th>
      <th>Imbalance</th>
    </tr>
  </thead>
  <tbody>
    {% for entry in datasets %}
      {% assign slug = entry[0] %}
      {% assign data = entry[1] %}
      {% assign dataset = data.dataset | default: data %}
      {% assign meta = dataset.meta %}
      {% assign name = meta.name | default: slug %}
      {% assign heading_id = name | replace: '_', ' ' | slugify %}
      {% assign instances = meta.instances | default: meta.sample_size | default: 'N/A' %}
      {% assign features = meta.features | default: meta.n_features | default: 'N/A' %}
      {% assign n_num = meta.n_num_features | default: 'N/A' %}
      {% assign n_cat = meta.n_cat_features | default: 'N/A' %}
      {% assign target = meta.target_variable | default: meta.target | default: 'N/A' %}
      {% assign n_classes = meta.n_classes | default: 'N/A' %}
      {% assign imbalance = meta.imbalance_ratio | default: 'N/A' %}
      <tr>
        <td><a href="#{{ heading_id }}">{{ name | replace: '_', ' ' | capitalize }}</a></td>
        <td>{{ instances }}</td>
        <td>{{ features }}</td>
        <td>{{ n_num }}</td>
        <td>{{ n_cat }}</td>
        <td>{{ target }}</td>
        <td>{{ n_classes }}</td>
        <td>{{ imbalance }}</td>
      </tr>
    {% endfor %}
  </tbody>
</table>
</div>




## Quick Check

<div class="table-scroll">
<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Type</th>
      <th>Family / Variant</th>
      <th>Test accuracy</th>
      <th>Test ROC AUC</th>
    </tr>
  </thead>
  <tbody>
    {% assign qc = site.data.quick_check.binary %}
    {% if qc %}
      {% for item in qc %}
        {% assign bc_pm = item.best_classical.primary_mean %}
        {% assign bt_pm = item.best_transformed.primary_mean %}
        {% assign bc_sm = item.best_classical.secondary_mean %}
        {% assign bt_sm = item.best_transformed.secondary_mean %}
        <tr>
          <td rowspan="2">{{ item.dataset | replace: '_', ' ' | capitalize }}</td>
          <td>Best classical</td>
          <td>{{ item.best_classical.family }} / {{ item.best_classical.best_variant }}</td>
          <td>
            {% if bc_pm != nil and bt_pm != nil %}
              {% if bc_pm >= bt_pm %}
                <strong>{{ item.best_classical.primary_raw | default: item.best_classical.primary_mean }}</strong>
              {% else %}
                {{ item.best_classical.primary_raw | default: item.best_classical.primary_mean }}
              {% endif %}
            {% else %}
              {{ item.best_classical.primary_raw | default: item.best_classical.primary_mean | default: 'N/A' }}
            {% endif %}
          </td>
          <td>
            {% if bc_sm != nil and bt_sm != nil %}
              {% if bc_sm >= bt_sm %}
                <strong>{{ item.best_classical.secondary_raw | default: item.best_classical.secondary_mean }}</strong>
              {% else %}
                {{ item.best_classical.secondary_raw | default: item.best_classical.secondary_mean }}
              {% endif %}
            {% else %}
              {{ item.best_classical.secondary_raw | default: item.best_classical.secondary_mean | default: 'N/A' }}
            {% endif %}
          </td>
        </tr>
        <tr>
          <td>Best transformed</td>
          <td>{{ item.best_transformed.family }} / {{ item.best_transformed.best_variant }}</td>
          <td>
            {% if bc_pm != nil and bt_pm != nil %}
              {% if bt_pm > bc_pm %}
                <strong>{{ item.best_transformed.primary_raw | default: item.best_transformed.primary_mean }}</strong>
              {% else %}
                {{ item.best_transformed.primary_raw | default: item.best_transformed.primary_mean }}
              {% endif %}
            {% else %}
              {{ item.best_transformed.primary_raw | default: item.best_transformed.primary_mean | default: 'N/A' }}
            {% endif %}
          </td>
          <td>
            {% if bc_sm != nil and bt_sm != nil %}
              {% if bt_sm > bc_sm %}
                <strong>{{ item.best_transformed.secondary_raw | default: item.best_transformed.secondary_mean }}</strong>
              {% else %}
                {{ item.best_transformed.secondary_raw | default: item.best_transformed.secondary_mean }}
              {% endif %}
            {% else %}
              {{ item.best_transformed.secondary_raw | default: item.best_transformed.secondary_mean | default: 'N/A' }}
            {% endif %}
          </td>
        </tr>
      {% endfor %}
    {% else %}
      <tr><td colspan="5">Quick check summaries not available. Generate `docs/_data/quick_check/binary.yml`.</td></tr>
    {% endif %}
  </tbody>
</table>
</div>

{% for entry in datasets %}
{% assign slug = entry[0] %}
{% assign data = entry[1] %}
{% assign dataset = data.dataset | default: data %}
{% assign meta = dataset.meta %}
{% assign name = meta.name | default: slug %}
{% assign heading_id = name | replace: '_', ' ' | slugify %}

## <a id="{{ heading_id }}"></a>{{ name | replace: '_', ' ' | capitalize }}

{% if meta.task_intro %}
{{ meta.task_intro | markdownify }}
{% elsif meta.source %}
<p><strong>Source:</strong> <a href="{{ meta.source }}">{{ meta.source }}</a></p>
{% endif %}

### Leaderboard

{% assign columns = dataset.leaderboard_columns %}
{% assign rows = dataset.leaderboard_rows %}
{% if rows and rows.size > 0 %}
<div class="table-scroll">
<table>
  <thead>
    <tr>
      {% for column in columns %}
        <th>{{ column.label }}</th>
      {% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for row in rows %}
      <tr>
        {% for column in columns %}
          {% assign key = column.key %}
          {% assign lower_is_better = false %}
          {% if key contains 'rmse' or key contains 'mae' or key contains 'mse' or key contains 'loss' or key contains 'time' or key contains 'training_time' or key contains 'params' or key contains 'flops' %}
            {% assign lower_is_better = true %}
          {% endif %}
          {% assign best_num = nil %}
          {% for r in rows %}
            {% assign rraw = r[key] | default: '' %}
            {% assign rtext = rraw | replace: ',', '' | replace: '%', '' | split: '±' | first | strip %}
            {% capture rdigits %}{{ rtext | replace: ' ', '' | replace: '0', '' | replace: '1', '' | replace: '2', '' | replace: '3', '' | replace: '4', '' | replace: '5', '' | replace: '6', '' | replace: '7', '' | replace: '8', '' | replace: '9', '' | replace: '.', '' | replace: '-', '' | replace: '+', '' | replace: 'e', '' | replace: 'E', '' }}{% endcapture %}
            {% if rtext != '' and rtext != '—' and rtext != 'N/A' and rdigits == '' %}
              {% assign rval = rtext | plus: 0 %}
              {% if best_num == nil %}
                {% assign best_num = rval %}
              {% else %}
                {% if lower_is_better %}
                  {% if rval < best_num %}{% assign best_num = rval %}{% endif %}
                {% else %}
                  {% if rval > best_num %}{% assign best_num = rval %}{% endif %}
                {% endif %}
              {% endif %}
            {% endif %}
          {% endfor %}
          {% assign raw = row[key] | default: 'N/A' %}
          {% assign value_text = raw | replace: ',', '' | replace: '%', '' | split: '±' | first | strip %}
          {% capture digits %}{{ value_text | replace: ' ', '' | replace: '0', '' | replace: '1', '' | replace: '2', '' | replace: '3', '' | replace: '4', '' | replace: '5', '' | replace: '6', '' | replace: '7', '' | replace: '8', '' | replace: '9', '' | replace: '.', '' | replace: '-', '' | replace: '+', '' | replace: 'e', '' | replace: 'E', '' }}{% endcapture %}
          {% assign should_bold = false %}
          {% if value_text != '' and value_text != '—' and value_text != 'N/A' and digits == '' and best_num != nil %}
            {% assign val = value_text | plus: 0 %}
            {% if val == best_num %}
              {% assign should_bold = true %}
            {% endif %}
          {% endif %}
          <td>{% if should_bold %}<strong>{{ raw }}</strong>{% else %}{{ raw }}{% endif %}</td>
        {% endfor %}
      </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% else %}
<i>No baseline leaderboard available.</i>
{% endif %}

<details class="arch-results">
  <summary>Architecture Results</summary>

  {% for section in dataset.arch_sections %}
  <details class="arch-section">
    <summary>{{ section.title }}</summary>

    {% assign section_columns = section.columns %}
    {% assign section_rows = section.rows %}
    {% if section_rows and section_rows.size > 0 %}
    <div class="table-scroll">
    <table>
      <thead>
        <tr>
          {% for column in section_columns %}
            <th>{{ column.label }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in section_rows %}
          <tr>
            {% for column in section_columns %}
              {% assign key = column.key %}
              {% assign lower_is_better = false %}
              {% if key contains 'rmse' or key contains 'mae' or key contains 'mse' or key contains 'loss' or key contains 'time' or key contains 'training_time' or key contains 'params' or key contains 'flops' %}
                {% assign lower_is_better = true %}
              {% endif %}
              {% assign best_num = nil %}
              {% for r in section_rows %}
                {% assign rraw = r[key] | default: '' %}
                {% assign rtext = rraw | replace: ',', '' | replace: '%', '' | split: '±' | first | strip %}
                {% capture rdigits %}{{ rtext | replace: ' ', '' | replace: '0', '' | replace: '1', '' | replace: '2', '' | replace: '3', '' | replace: '4', '' | replace: '5', '' | replace: '6', '' | replace: '7', '' | replace: '8', '' | replace: '9', '' | replace: '.', '' | replace: '-', '' | replace: '+', '' | replace: 'e', '' | replace: 'E', '' }}{% endcapture %}
                {% if rtext != '' and rtext != '—' and rtext != 'N/A' and rdigits == '' %}
                  {% assign rval = rtext | plus: 0 %}
                  {% if best_num == nil %}
                    {% assign best_num = rval %}
                  {% else %}
                    {% if lower_is_better %}
                      {% if rval < best_num %}{% assign best_num = rval %}{% endif %}
                    {% else %}
                      {% if rval > best_num %}{% assign best_num = rval %}{% endif %}
                    {% endif %}
                  {% endif %}
                {% endif %}
              {% endfor %}
              {% assign raw = row[key] | default: 'N/A' %}
              {% assign value_text = raw | replace: ',', '' | replace: '%', '' | split: '±' | first | strip %}
              {% capture digits %}{{ value_text | replace: ' ', '' | replace: '0', '' | replace: '1', '' | replace: '2', '' | replace: '3', '' | replace: '4', '' | replace: '5', '' | replace: '6', '' | replace: '7', '' | replace: '8', '' | replace: '9', '' | replace: '.', '' | replace: '-', '' | replace: '+', '' | replace: 'e', '' | replace: 'E', '' }}{% endcapture %}
              {% assign should_bold = false %}
              {% if value_text != '' and value_text != '—' and value_text != 'N/A' and digits == '' and best_num != nil %}
                {% assign val = value_text | plus: 0 %}
                {% if val == best_num %}
                  {% assign should_bold = true %}
                {% endif %}
              {% endif %}
              <td>{% if should_bold %}<strong>{{ raw }}</strong>{% else %}{{ raw }}{% endif %}</td>
            {% endfor %}
          </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
    {% else %}
    <i>No results available for this architecture.</i>
    {% endif %}
  </details>
  {% endfor %}

</details>

---
{% endfor %}
