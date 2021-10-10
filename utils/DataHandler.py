import enum

relation0 = ['Attribution', 'Enablement', 'Elaboration', 'Joint', 'Same-Unit', 'Other']
relation1 = ['Background', 'Cause', 'Comparison', 'Contrast', 'Condition', 'Evaluation', 'Explanation',
                'Manner-Means','Topic-Comment', 'Summary', 'Temporal', 'Topic-Change', 'Textual-Organization']


relation_map0 ={rel:i for rel,i in zip(relation0, range(len(relation0)))}
relation_map1 ={rel:i for rel,i in zip(relation1, range(len(relation1)))}

index_2_relation0 ={aa:bb for bb,aa in relation_map0.items()}
index_2_relation1 = {aa:bb for bb,aa in relation_map1.items()}

form_map0 = {'NN':0, 'NS':1, 'SN':2, 'N~':3 }


index_2_form0 = {0: 'NN', 1:'NS', 2:'SN', 3:'N~'}



def find_type(token_dict, text):
    span_type = ['0'] * 4
    if text[0] - 1 < 0 or token_dict[text[0] - 1].sidx < token_dict[text[0]].sidx:
        span_type[0] = '1'

    # start of a paragraph
    if (text[0] - 1 < 0 or token_dict[text[0] - 1].pidx < token_dict[text[0]].pidx):
        span_type[1] = '1'


    # end of sentence
    if text[-1]+1 not in token_dict or token_dict[text[-1]].sidx < token_dict[text[-1]+1].sidx:
        span_type[2] = '1'


    # end of paragraph
    if text[-1]+1 not in token_dict or token_dict[text[-1]].pidx < token_dict[text[-1]+1].pidx:
        span_type[3] = '1'

    return int(''.join(span_type), 2)




def get_RelationAndNucleus(form, relation, start_span, end_span, cut):

        label = []
        if form == 'NN' or form == 'N~':
            label = [((start_span, cut), "Nucleus", relation),
                     ((cut+1, end_span), "Nucleus", relation)]
        elif form == 'NS':
            label = [((start_span, cut), "Nucleus", 'span'),
                     ((cut+1, end_span), "Satellite", relation)]
        elif form == 'SN':
            label = [((start_span, cut), "Satellite", relation),
                     ((cut+1, end_span), "Nucleus", 'span')]
        return label


class Action(enum.Enum):
    Shift = 0
    Reduce = 1
