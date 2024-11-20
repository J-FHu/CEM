"""Utility functions for results visualization."""

import cv2
import base64


def get_sortable_html_header(column_name_list, sort_by_ascending=False):
  """Gets header for sortable html page.

  Basically, the html page contains a sortable table, where user can sort the
  rows by a particular column by clicking the column head.

  Example:

  column_name_list = [name_1, name_2, name_3]
  header = get_sortable_html_header(column_name_list)
  footer = get_sortable_html_footer()
  sortable_tabel = ...
  html_page = header + sortable_tabel + footer

  Args:
    column_name_list: List of column header names.
    sort_by_ascending: Default sorting order. If set as `True`, the html page
      will be sorted by ascending order when the header is clicked for the first
      time.

  Returns:
    A string, which represents for the header for a sortable html page.
  """
  header = '\n'.join([
    '<script type="text/javascript">',
    'var column_index;',
    'var sort_by_ascending = %s;' % ('true' if sort_by_ascending else 'false'),
    'function sorting(tbody, column_index){',
    '  this.column_index = column_index;',
    '  Array.from(tbody.rows)',
    '       .sort(compareCells)',
    '       .forEach(function(row) { tbody.appendChild(row); })',
    '  sort_by_ascending = !sort_by_ascending;',
    '}',
    'function compareCells(row_a, row_b) {',
    '  var val_a = row_a.cells[column_index].innerText;',
    '  var val_b = row_b.cells[column_index].innerText;',
    '  var flag = sort_by_ascending ? 1 : -1;',
    '  return flag * (val_a > val_b ? 1 : -1);',
    '}',
    'window.onscroll = function()',
    '{',
    'var fix = document.getElementsByName("fix");',
    'var t = document.body.scrollLeft;',
    'for (var i=0;i < fix.length;i++)',
    '{',
    'fix[i].style.left = t + "px";',
    '}',
    '}',
    '</script>',
    '',
    '<html>',
    '',
    '<head>',
    '<style>',
    '  table {',
    '    border-spacing: 0;',
    '    border: 1px solid black;',
    '  }',
    '  th {',
    '    cursor: pointer;',
    '  }',
    '  th, td {',
    '    text-align: left;',
    '    vertical-align: middle;',
    '    border-collapse: collapse;',
    '    border: 0.5px solid black;',
    '    padding: 8px;',
    '  }',
    '  tr:nth-child(even) {',
    '    background-color: #d2d2d2;',
    '  }',
    '</style>',
    '</head>',
    '',
    '<body>',
    '',
    '<table>',
    '<thead>',
    '<tr>',
    ''])
  for idx, column_name in enumerate(column_name_list):
    header += '  <th onclick="sorting(tbody, {})">{}</th>\n'.format(
        idx, column_name)
  header += '</tr>\n'
  header += '</thead>\n'
  header += '<tbody id="tbody">\n'

  return header


def get_sortable_html_footer():
  """Gets footer for sortable html page.

  Check function `get_sortable_html_header()` for more details.
  """
  return '</tbody>\n</table>\n\n</body>\n</html>\n'


def convert_image_to_html(image):
  """Converts an image to html language for visualization."""
  encoded_image = cv2.imencode(".jpg", image)[1].tostring()
  encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
  html = ('<img src="data:image/jpeg;base64, ' + encoded_image_base64 +
          '"/>')
  return html

def convert_image_to_html_fixed_left(image):
  """Converts an image to html language for visualization."""
  encoded_image = cv2.imencode(".jpg", image)[1].tostring()
  encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
  html = ('<img src="data:image/jpeg;base64, ' + encoded_image_base64 +
          '"/>')
  return html
