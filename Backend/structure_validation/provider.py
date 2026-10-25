import torch
import torch.nn as nn
from pydantic import BaseModel,Field
from typing import Annotated

# classifier architecture
class FraudDetector(nn.Module):
    def __init__(self,inp_dim):
        super().__init__()
        self.network= nn.Sequential(
                    nn.Linear(inp_dim,256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.2),

                    nn.Linear(256,128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.1),

                    nn.Linear(128,64),
                    nn.BatchNorm1d(64),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.1),

                    nn.Linear(64,1),
                    nn.Sigmoid())
    def forward(self,features):
        return self.network(features)


# request data validator
class Validation(BaseModel):
    employment_type:Annotated[int,Field(...,gt=-1,lt=4,title='Employment type as a label ',description='labels are mapped as -> Internship: 0, Contract: 1, Part-Time: 2, Full-Time: 3 ')]

    company_size:Annotated[int,Field(...,gt=-1,lt=4,title='Company size as an integer label ',description='labels are mapped as -> Startup: 0, Small: 1, Medium: 2, Enterprise: 3 ')]

    company_age:Annotated[int,Field(...,gt=-1,title='Age of the company')]

    linkedin_presence:Annotated[int,Field(...,gt=-1,lt=2,title='Presence of the company on LinkedIn',description='Requires a boolean integer value, True: 1, False: 0')]

    website_available:Annotated[int,Field(...,gt=-1,lt=2,title='Is there any website of the company',description='Requires a boolean integer value, True: 1, False: 0')]

    verification_status:Annotated[int,Field(...,gt=-1,lt=2,title='Is the company present on LinkedIn',description='Requires a boolean integer value, True: 1, False: 0')]

    stipend:Annotated[float,Field(...,gt=-1,title='How much stipend is provided',description='Provide the stipend if any, else enter -> 0')]

    registration_fee:Annotated[float,Field(...,gt=-1,title='How much is the registration fee',description='Provide the fee if any, else enter -> 0')]

    vague_description_score:Annotated[float,Field(...,gt=-1,lt=101,title='Description stating Unclear words',description='How many words are unclear out of total')]

    urgency_score:Annotated[float,Field(...,gt=-1,lt=101,title='Description stating urgency',description='How many words do you think want immense attention out of total')]

    keyword_spam_score:Annotated[float,Field(...,gt=-1,lt=101,title='Description having spam words',description='How many words are suspicious out of total')]

    suspicious_email_domain:Annotated[int,Field(...,gt=-1,lt=2,title='Is the email domain suspicious',description='Requires a boolean integer value, True: 1, False: 0')]
