import sqlalchemy

def _col_type_name(col) -> str:
    t = col.type
    if isinstance(t, sqlalchemy.Integer):
        return 'integer'
    if isinstance(t, (sqlalchemy.String, sqlalchemy.Text)):
        return 'string'
    if isinstance(t, sqlalchemy.Float):
        return 'float'
    if isinstance(t, sqlalchemy.Boolean):
        return 'boolean'
    if isinstance(t, sqlalchemy.DateTime):
        return 'datetime'
    if isinstance(t, sqlalchemy.Date):
        return 'date'
    if isinstance(t, sqlalchemy.Time):
        return 'time'
    if isinstance(t, sqlalchemy.LargeBinary):
        return 'bytes'
    if isinstance(t, sqlalchemy.JSON):
        return 'dict'
    if isinstance(t, sqlalchemy.ARRAY):
        return 'list'
    return 'text'
