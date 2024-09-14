---
layout: default
title: My Github projects.
---

[Return to index](/)

<ul>
  {% for gh in site.collections.github %}
    <li>
      <a href="{{ gh.url }}">{{ gh.title }}</a>
    </li>
  {% endfor %}
</ul>


{% for staff_member in site.staff_members %}
  <h2>{{ staff_member.name }} - {{ staff_member.position }}</h2>
  <p>{{ staff_member.content | markdownify }}</p>
{% endfor %}