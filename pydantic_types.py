from typing import List, Optional, Type

from pydantic import BaseModel, ConfigDict, conset

from json_schema_to_grammar import SchemaConverter


class Emissions(BaseModel):
     model_config = ConfigDict(strict=True)
     scope_1: Optional[int]
     scope_2: Optional[int]
     scope_3: Optional[int]
     sources: conset(int, min_length=0, max_length=5)


def pydantic_model_to_grammar(model_type: Type[BaseModel]):
    schema = model_type.model_json_schema()
    converter = SchemaConverter({})
    converter.visit(schema, "")
    grammar = converter.format_grammar()
    return grammar
