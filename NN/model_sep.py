"""
    author: Jinfen Li
    GitHub: https://github.com/LiJinfen
"""

import torch
import torch.nn as nn
from module import SpanEncoder, SplittingModel, FormModel0,RelModel0, RelModel1
from utils import DataHandler


class ParsingNet(nn.Module):


    def __init__(self, elmo,  device, hidden_size=200, type_num=16, type_dim=10):

        super(ParsingNet, self).__init__()
        self.span_encoder = SpanEncoder(elmo, device, hidden_size=hidden_size)
        self.splitting_model = SplittingModel(hidden_size)
        self.form_model0 = FormModel0(hidden_size)
        self.rel_model0 = RelModel0(hidden_size)
        self.rel_model1 = RelModel1(hidden_size)
        self.device = device
        self.type_embeddings = nn.Embedding(type_num, type_dim)


    def TrainingLoss(self, train_exps_batch):
        '''
            Input:
                train_exps_batch: label: {(1, 69): ('NN', 'Textual-Organization', 1), (2, 69): ('NS', 'Elaboration', 3), (4, 69): ('NN', 'Joint', 6), (7, 69): ('NN', 'Joint', 14),...}
                                text:{(1, 1): 'text ...', ...}
            Output:
                Average loss of split, form, relation in a batch
        '''

        total_loss_action = 0
        total_loss_form = 0
        total_loss_rel = 0
        total_loss_rel1 = 0
        loop_form = 0
        loop_action = 0
        loop_rel = 0
        loop_rel1 = 0
        loss = nn.CrossEntropyLoss()


        for exp_id in range(len(train_exps_batch)):

            train_exps = train_exps_batch[exp_id]
            token_dict = train_exps.token_dict
            texts = train_exps.texts

            cur_text = []

            spans = sorted(texts.keys())
            for unitId, k in enumerate(spans):
                cur_text.append(texts[k]['text'])

            labels = train_exps.label
            # batch(span_num) x 1 x hidden
            f, b = self.span_encoder(cur_text)
            min_id = spans[0][0]
            max_id = spans[-1][1]
            stack = [(min_id, min_id)]

            i = 2

            while stack!=[(min_id, max_id)]:
                # reduce now
                if len(stack)>1:
                    start = stack[-2][0]
                    cut = stack[-2][1]
                    end = stack[-1][1]

                    if (start, end) in labels:
                        action = DataHandler.Action.Reduce.value
                        stack = stack[:-2]
                        stack.append((start, end))


                    else:
                        action = DataHandler.Action.Shift.value
                        stack.append((i, i))
                        i += 1


                    left_type = DataHandler.find_type(token_dict,
                                                      texts[(start, start)]['token_ids'] + texts[(cut, cut)][
                                                          'token_ids'])

                    right_type = DataHandler.find_type(token_dict,
                                                       texts[(cut+1, cut+1)]['token_ids'] +
                                                       texts[(end, end)][
                                                           'token_ids'])
                    left_type_embs = self.type_embeddings(torch.tensor(left_type).to(self.device)).unsqueeze(0)
                    right_type_embs = self.type_embeddings(torch.tensor(right_type).to(self.device)).unsqueeze(0)
                    pred_action = self.splitting_model(start=start,
                                                      end=end, cut=cut, f=f, b=b,
                                                      left_type_embs=left_type_embs, right_type_embs=right_type_embs)


                    loss_action = loss(pred_action, torch.tensor([action]).to(self.device))
                    loop_action += 1
                    total_loss_action += loss_action

                    if action == DataHandler.Action.Reduce.value:
                        relation = labels[stack[-1]][1]

                        form_score = self.form_model0(start=start, end=end,
                                                                    cut0=cut,cut1=cut+1, f=f, b=b, left_type_embs=left_type_embs,
                                                                    right_type_embs=right_type_embs)


                        form = labels[stack[-1]][0]
                        loss_form = loss(form_score, torch.tensor([DataHandler.form_map0[form]]).to(self.device))
                        loop_form += 1
                        total_loss_form += loss_form
                        trans_relation1 = relation

                        if relation not in DataHandler.relation_map0:
                            trans_relation1 = 'Other'

                        pred_form = DataHandler.form_map0[form]
                        rel_score = self.rel_model0(start=start, end=end,
                                                    cut0=cut, cut1=cut + 1, f=f, b=b, left_type_embs=left_type_embs,
                                                    right_type_embs=right_type_embs, form=pred_form)
                        loss_rel = loss(rel_score,
                                        torch.tensor([DataHandler.relation_map0[trans_relation1]]).to(self.device))


                        loop_rel += 1
                        total_loss_rel += loss_rel

                        if trans_relation1 == 'Other':
                            rel_score = self.rel_model1(start=start, end=end,
                                                        cut0=cut, cut1=cut + 1, f=f, b=b, left_type_embs=left_type_embs,
                                                        right_type_embs=right_type_embs, form=pred_form)
                            loss_rel = loss(rel_score,
                                            torch.tensor([DataHandler.relation_map1[relation]]).to(self.device))

                            loop_rel1 += 1
                            total_loss_rel1 += loss_rel


                else:
                    stack.append((i, i))
                    i += 1

        return total_loss_action/loop_action, total_loss_form/loop_form, total_loss_rel/loop_rel if loop_rel!=0 else 0,total_loss_rel1/loop_rel1 if loop_rel1!=0 else 0


    def TestingLoss(self, test_exp, is_predict=False):

        '''
        Input:
            train_exps_batch: label: {(1, 69): ('NN', 'Textual-Organization', 1), (2, 69): ('NS', 'Elaboration', 3), (4, 69): ('NN', 'Joint', 6), (7, 69): ('NN', 'Joint', 14),...}
                            text:{(1, 1): 'text ...', ...}
        Output: 
            Average loss of split, form, relation in a batch
        '''

        pred_batch = []
        token_dict = test_exp.token_dict
        texts = test_exp.texts
        cur_text = []
        spans = sorted(texts.keys())
        for unitId, k in enumerate(spans):
            cur_text.append(texts[k]['text'])

        # batch x seq x hidden, rnn_layer x batch x hidden
        f, b = self.span_encoder(cur_text)

        min_id = spans[0][0]
        max_id = spans[-1][1]
        stack = [(min_id, min_id)]
        depth = {}
        for j in range(min_id, max_id+1):
            depth[(j,j)] = 1
        i = 2
        while stack != [(min_id, max_id)]:
            # reduce now
            if len(stack) > 1:
                start = stack[-2][0]
                cut = stack[-2][1]
                end = stack[-1][1]
                left_type = DataHandler.find_type(token_dict,
                                                  texts[(start, start)]['token_ids'] + texts[(cut, cut)][
                                                      'token_ids'])
                right_type = DataHandler.find_type(token_dict,
                                                   texts[(cut + 1, cut + 1)]['token_ids'] +
                                                   texts[(end, end)][
                                                       'token_ids'])
                left_type_embs = self.type_embeddings(torch.tensor(left_type).to(self.device)).unsqueeze(0)
                right_type_embs = self.type_embeddings(torch.tensor(right_type).to(self.device)).unsqueeze(0)

                pred_action = self.splitting_model(start=start,
                                                   end=end, cut=cut, f=f, b=b,
                                                   left_type_embs=left_type_embs, right_type_embs=right_type_embs)

                if end == max_id or torch.argmax(pred_action).item() == DataHandler.Action.Reduce.value:
                    action = DataHandler.Action.Reduce.value
                    stack = stack[:-2]
                    stack.append((start, end))
                else:
                    action = DataHandler.Action.Shift.value
                    stack.append((i, i))
                    i += 1

                if action == DataHandler.Action.Reduce.value:

                    form_score = self.form_model0(start=start, end=end,
                                                 cut0=cut, cut1=cut+1,  f=f, b=b, left_type_embs=left_type_embs,
                                                      right_type_embs=right_type_embs)


                    pred_form = torch.argmax(form_score).item()

                    rel_score = self.rel_model0(start=start, end=end,
                                                cut0=cut, cut1=cut+1, f=f, b=b, left_type_embs=left_type_embs,
                                                right_type_embs=right_type_embs, form=pred_form)


                    pred_rel = torch.argmax(rel_score).item()
                    if pred_rel == len(DataHandler.relation_map0)-1:
                        rel_score = self.rel_model1(start=start, end=end,
                                                    cut0=cut, cut1=cut + 1, f=f, b=b, left_type_embs=left_type_embs,
                                                    right_type_embs=right_type_embs, form=pred_form)

                        pred_rel = torch.argmax(rel_score).item()
                        pred_rel = DataHandler.index_2_relation1[pred_rel]

                    else:
                        pred_rel = DataHandler.index_2_relation0[pred_rel]
                    pred_form = DataHandler.index_2_form0[pred_form]
                    label = DataHandler.get_RelationAndNucleus(pred_form, pred_rel, start, end, cut)
                    pred_batch.extend(label)
                    depth[(start,end)] = max(depth[(start, cut)],depth[(cut+1, end)])+1
                    # depth[(cut+1, end)] = depth[(start, end)] + 1

            else:
                stack.append((i, i))
                i += 1

        return pred_batch, depth