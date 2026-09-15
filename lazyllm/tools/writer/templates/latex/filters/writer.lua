-- LazyMind Markdown contract for Markdown-to-LaTeX conversion.
-- The first and only H1 is the document title. Body headings start at H2 and
-- are promoted by one level so H2 becomes a LaTeX section.

local function task_marker(item)
  local first_block = item[1]
  if first_block == nil or
      (first_block.t ~= 'Plain' and first_block.t ~= 'Para') or
      #first_block.content == 0 then
    return nil
  end
  local first_inline = first_block.content[1]
  if first_inline.t ~= 'Str' then
    return nil
  end
  if first_inline.text == '☐' then
    return '\\taskunchecked'
  end
  if first_inline.text == '☒' then
    return '\\taskchecked'
  end
  return nil
end

local function render_task_list(list)
  local items = list.content
  local markers = {}
  for index, item in ipairs(items) do
    markers[index] = task_marker(item)
    if markers[index] == nil then
      return nil
    end
  end

  local output = {'\\begin{itemize}[leftmargin=*]'}
  for index, item in ipairs(items) do
    local content = item[1].content
    content:remove(1)
    if content[1] ~= nil and content[1].t == 'Space' then
      content:remove(1)
    end
    local body = pandoc.write(pandoc.Pandoc(pandoc.Blocks(item)), 'latex'):gsub('%s+$', '')
    table.insert(output, '\\item[' .. markers[index] .. '] ' .. body)
  end
  table.insert(output, '\\end{itemize}')
  return pandoc.RawBlock('latex', table.concat(output, '\n'))
end

local function degrade_special_code_block(block)
  for _, class in ipairs(block.classes) do
    if class == 'mermaid' then
      block.classes = pandoc.List({})
      block.attributes = {}
      return block
    end
  end
  return nil
end

local function html_attribute(opening, name)
  local value = opening:match('%f[%w]' .. name .. '%s*=%s*"([^"]*)"')
  if value == nil then
    value = opening:match("%f[%w]" .. name .. "%s*=%s*'([^']*)'")
  end
  return value
end

local function html_unescape(value)
  if value == nil then
    return nil
  end
  value = value:gsub('&#x27;', "'"):gsub('&#39;', "'"):gsub('&quot;', '"')
  value = value:gsub('&lt;', '<'):gsub('&gt;', '>'):gsub('&amp;', '&')
  return value
end

local function empty_anchor_opening(block)
  if block.t ~= 'Para' or #block.content ~= 2 then
    return nil
  end
  local opening = block.content[1]
  local closing = block.content[2]
  if opening.t ~= 'RawInline' or opening.format ~= 'html' or
      closing.t ~= 'RawInline' or closing.format ~= 'html' or
      not opening.text:match('^<a%s+[^>]*>%s*$') or
      not closing.text:match('^</a>%s*$') then
    return nil
  end
  return opening.text
end

local function latex_text(value)
  return pandoc.write(
    pandoc.Pandoc(pandoc.Blocks({
      pandoc.Plain(pandoc.Inlines({pandoc.Str(value)}))
    })), 'latex'
  ):gsub('%s+$', '')
end

local function latex_block(block)
  return pandoc.write(
    pandoc.Pandoc(pandoc.Blocks({block})), 'latex'
  ):gsub('%s+$', '')
end

local function render_numbered_bibliography(doc)
  local blocks = pandoc.Blocks({})
  local keys = {}
  local index = 1
  while index <= #doc.blocks do
    local opening = empty_anchor_opening(doc.blocks[index])
    local marker = opening and html_attribute(
      opening, 'data%-writer%-bibliography'
    ) or nil
    if marker ~= 'begin' then
      blocks:insert(doc.blocks[index])
      index = index + 1
    else
      local title = html_unescape(html_attribute(opening, 'data%-title')) or ''
      local items = {}
      index = index + 1
      while index <= #doc.blocks do
        local item_opening = empty_anchor_opening(doc.blocks[index])
        local key = item_opening and html_attribute(
          item_opening, 'data%-writer%-bibitem'
        ) or nil
        if key == nil or not key:match('^[A-Za-z0-9_.:%-]+$') or
            doc.blocks[index + 1] == nil then
          error('LazyMind Markdown bibliography marker has no valid item')
        end
        if keys[key] then
          error('LazyMind Markdown bibliography contains duplicate key: ' .. key)
        end
        keys[key] = true
        table.insert(items, {
          key = key,
          body = latex_block(doc.blocks[index + 1]),
        })
        index = index + 2
      end
      local widest = string.rep('9', math.max(1, #tostring(#items)))
      local output = {'{'}
      if title ~= '' then
        table.insert(output, '\\renewcommand{\\refname}{' .. latex_text(title) .. '}')
      end
      table.insert(output, '\\begin{thebibliography}{' .. widest .. '}')
      for _, item in ipairs(items) do
        table.insert(output, '\\bibitem{' .. item.key .. '}')
        table.insert(output, item.body)
      end
      table.insert(output, '\\end{thebibliography}')
      table.insert(output, '}')
      blocks:insert(pandoc.RawBlock('latex', table.concat(output, '\n')))
    end
  end
  doc.blocks = blocks
  return doc, keys
end

local function rewrite_bibliography_citations(doc, keys)
  return doc:walk({
    Link = function(link)
      local key = link.target:match('^#writer%-cite%-(ref%-%d+)$')
      if key == nil then
        return nil
      end
      if not keys[key] then
        pandoc.log.warn(
          'LazyMind Markdown bibliography target does not exist: ' .. key
        )
        return link.content
      end
      return pandoc.RawInline('latex', '\\cite{' .. key .. '}')
    end
  })
end

local function anchor_prefix(block)
  if block.t ~= 'Para' or #block.content < 2 then
    return nil
  end
  local opening = block.content[1]
  local closing = block.content[2]
  if opening.t ~= 'RawInline' or opening.format ~= 'html' or
      closing.t ~= 'RawInline' or closing.format ~= 'html' or
      not closing.text:match('^</a>%s*$') then
    return nil
  end
  if not opening.text:match('^<a%s+[^>]*>%s*$') then
    return nil
  end
  local id = html_attribute(opening.text, 'id')
  if id == nil or not id:match('^block%-[A-Za-z0-9_.:%-]+$') then
    return nil
  end
  local declared_kind = html_attribute(opening.text, 'data%-kind')
  if declared_kind ~= nil then
    declared_kind = declared_kind:lower()
  end
  local caption_text = html_unescape(html_attribute(opening.text, 'data%-caption'))
  local caption = nil
  if caption_text ~= nil then
    caption = pandoc.Inlines({pandoc.Str(caption_text)})
  end

  local target = nil
  if #block.content > 2 then
    local start = 3
    if block.content[start] ~= nil and block.content[start].t == 'SoftBreak' then
      start = start + 1
    end
    local content = pandoc.Inlines({})
    for index = start, #block.content do
      content:insert(block.content[index])
    end
    if #content > 0 then
      target = pandoc.Para(content)
    end
  end
  return id, target, declared_kind, caption
end

local function target_kind(block)
  if block == nil then
    return nil
  end
  if block.t == 'Header' then
    return 'section'
  end
  if block.t == 'Table' then
    return 'table'
  end
  if block.t == 'CodeBlock' then
    return 'code'
  end
  if block.t == 'Para' and #block.content == 1 then
    if block.content[1].t == 'Image' then
      return 'figure'
    end
    if block.content[1].t == 'Math' and block.content[1].mathtype == 'DisplayMath' then
      return 'equation'
    end
  end
  return nil
end

local function image_asset_path(source)
  local normalized = source:gsub('\\', '/')
  local filename = normalized:match('([^/]+)$')
  if filename == nil or filename == '' then
    error('LazyMind Markdown image source has no filename: ' .. source)
  end
  return 'assets/' .. filename
end

local function render_figure(block, label, anchor_caption)
  local image = block.content[1]
  image.src = image_asset_path(image.src)
  local caption = anchor_caption or image.caption
  image.caption = pandoc.Inlines({})
  local identifier = label or ''
  return pandoc.Figure(
    pandoc.Blocks({pandoc.Plain({image})}),
    pandoc.Caption(pandoc.Blocks({pandoc.Plain(caption)})),
    pandoc.Attr(identifier)
  )
end

local function latex_label(id, kind)
  local prefixes = {
    section = 'sec',
    figure = 'fig',
    table = 'table',
    code = 'code',
    equation = 'eq',
  }
  return prefixes[kind] .. ':' .. id:sub(#'block-' + 1)
end

local function latex_inlines(inlines)
  return pandoc.write(
    pandoc.Pandoc(pandoc.Blocks({pandoc.Plain(inlines)})), 'latex'
  ):gsub('%s+$', '')
end

local function render_code_caption(caption, label)
  local text = latex_inlines(caption)
  local suffix = ''
  if text ~= '' then
    suffix = '\\codeblockcaptionseparator{}' .. text
  end
  return pandoc.RawBlock(
    'latex',
    '\\refstepcounter{codeblock}\\label{' .. label .. '}\n' ..
    '\\noindent\\textbf{\\codeblockname~\\thecodeblock' .. suffix .. '}\\par'
  )
end

-- Pandoc renders default-width Markdown columns as l/c/r columns, which do
-- not wrap and can extend past the page. Assign the remaining line width to
-- default columns so the LaTeX writer emits wrapping p{...} columns while
-- retaining longtable's page-breaking behavior. Explicit source widths are
-- preserved.
local function constrain_table_width(block)
  local default_count = 0
  local explicit_width = 0
  for _, colspec in ipairs(block.colspecs) do
    local width = colspec[2]
    if type(width) == 'number' and width > 0 then
      explicit_width = explicit_width + width
    else
      default_count = default_count + 1
    end
  end
  if default_count == 0 or explicit_width >= 1 then
    return block
  end
  local default_width = (1 - explicit_width) / default_count
  for _, colspec in ipairs(block.colspecs) do
    local width = colspec[2]
    if type(width) ~= 'number' or width <= 0 then
      colspec[2] = default_width
    end
  end
  return block
end

local function bind_anchor(blocks, block, id, labels, caption, declared_kind)
  local kind = target_kind(block)
  if kind == nil then
    pandoc.log.warn('LazyMind Markdown anchor has unsupported target: ' .. id)
    blocks:insert(block)
    return
  end
  if declared_kind ~= nil and declared_kind ~= kind then
    error(
      'LazyMind Markdown anchor kind does not match target: ' .. id ..
      ' declares ' .. declared_kind .. ', got ' .. kind
    )
  end

  local label = latex_label(id, kind)
  labels[id] = label
  if kind == 'section' then
    block.identifier = label
    blocks:insert(block)
  elseif kind == 'figure' then
    blocks:insert(render_figure(block, label, caption))
  elseif kind == 'table' then
    if caption ~= nil then
      block.caption = pandoc.Caption(pandoc.Blocks({pandoc.Plain(caption)}))
      block.identifier = label
    else
      blocks:insert(pandoc.RawBlock(
        'latex', '\\refstepcounter{table}\\label{' .. label .. '}'
      ))
    end
    blocks:insert(block)
  elseif kind == 'code' then
    if caption ~= nil then
      blocks:insert(render_code_caption(caption, label))
    else
      blocks:insert(pandoc.RawBlock(
        'latex', '\\refstepcounter{codeblock}\\label{' .. label .. '}'
      ))
    end
    blocks:insert(block)
  else
    local formula = block.content[1].text
    blocks:insert(pandoc.RawBlock(
      'latex', '\\begin{equation}\n' .. formula .. '\n\\label{' .. label .. '}\n\\end{equation}'
    ))
  end
end

local function bind_anchors(doc)
  local labels = {}
  local blocks = pandoc.Blocks({})
  local pending_id = nil
  local pending_kind = nil
  local pending_caption = nil
  local index = 1

  while index <= #doc.blocks do
    local block = doc.blocks[index]
    local id, inline_target, declared_kind, anchor_caption = anchor_prefix(block)
    if id ~= nil then
      if pending_id ~= nil then
        pandoc.log.warn('LazyMind Markdown anchor has no target: ' .. pending_id)
        pending_id = nil
        pending_kind = nil
        pending_caption = nil
      end
      if labels[id] ~= nil then
        error('LazyMind Markdown contains duplicate anchor: ' .. id)
      end
      if inline_target ~= nil then
        bind_anchor(blocks, inline_target, id, labels, anchor_caption, declared_kind)
      else
        pending_id = id
        pending_kind = declared_kind
        pending_caption = anchor_caption
      end
    else
      if pending_id ~= nil then
        bind_anchor(blocks, block, pending_id, labels, pending_caption, pending_kind)
        pending_id = nil
        pending_kind = nil
        pending_caption = nil
      else
        if block.t == 'Header' then
          block.identifier = ''
        end
        blocks:insert(block)
      end
    end
    index = index + 1
  end

  if pending_id ~= nil then
    pandoc.log.warn('LazyMind Markdown anchor has no target: ' .. pending_id)
  end
  doc.blocks = blocks
  return labels
end

local function rewrite_internal_references(doc, labels)
  return doc:walk({
    Link = function(link)
      local id = link.target:match('^#(block%-[A-Za-z0-9_.:%-]+)$')
      if id == nil then
        return nil
      end
      local target = labels[id]
      if target == nil then
        pandoc.log.warn(
          'LazyMind Markdown internal reference target does not exist: ' .. id
        )
        return link.content
      end
      local content = pandoc.Inlines({})
      for _, inline in ipairs(link.content) do
        content:insert(inline)
      end
      content:insert(pandoc.RawInline(
        'latex', '\\writerinternalref{' .. target .. '}'
      ))
      return content
    end
  })
end

function Pandoc(doc)
  local title = doc.blocks[1]
  if title == nil or title.t ~= 'Header' or title.level ~= 1 then
    error('LazyMind Markdown must start with exactly one H1 document title')
  end

  doc.meta.title = pandoc.MetaInlines(title.content)
  table.remove(doc.blocks, 1)

  local bibliography_keys
  doc, bibliography_keys = render_numbered_bibliography(doc)
  local labels = bind_anchors(doc)

  doc = doc:walk({
    Header = function(header)
      if header.level == 1 then
        error('LazyMind Markdown may contain only one H1 document title')
      end
      header.level = header.level - 1
      return header
    end,
    Para = function(para)
      if target_kind(para) == 'figure' then
        return render_figure(para)
      end
      return nil
    end,
    Table = constrain_table_width,
  })
  doc = rewrite_internal_references(doc, labels)
  doc = rewrite_bibliography_citations(doc, bibliography_keys)
  doc = doc:walk({
    BulletList = render_task_list,
    CodeBlock = degrade_special_code_block,
  })

  return doc
end
