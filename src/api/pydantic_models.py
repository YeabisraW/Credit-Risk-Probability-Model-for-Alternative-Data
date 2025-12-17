# src/api/pydantic_models.py
from pydantic import BaseModel

class CustomerData(BaseModel):
    # Numeric features
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    hour: int
    day: int
    month: int
    year: int

    # Categorical one-hot encoded features
    CurrencyCode_UGX: int

    ProviderId_ProviderId_1: int
    ProviderId_ProviderId_2: int
    ProviderId_ProviderId_3: int
    ProviderId_ProviderId_4: int
    ProviderId_ProviderId_5: int
    ProviderId_ProviderId_6: int

    ProductId_ProductId_1: int
    ProductId_ProductId_2: int
    ProductId_ProductId_3: int
    ProductId_ProductId_4: int
    ProductId_ProductId_5: int
    ProductId_ProductId_6: int
    ProductId_ProductId_7: int
    ProductId_ProductId_8: int
    ProductId_ProductId_9: int
    ProductId_ProductId_10: int
    ProductId_ProductId_11: int
    ProductId_ProductId_12: int
    ProductId_ProductId_13: int
    ProductId_ProductId_14: int
    ProductId_ProductId_15: int
    ProductId_ProductId_16: int
    ProductId_ProductId_19: int
    ProductId_ProductId_20: int
    ProductId_ProductId_21: int
    ProductId_ProductId_22: int
    ProductId_ProductId_24: int
    ProductId_ProductId_27: int

    ProductCategory_airtime: int
    ProductCategory_data_bundles: int
    ProductCategory_financial_services: int
    ProductCategory_movies: int
    ProductCategory_other: int
    ProductCategory_ticket: int
    ProductCategory_transport: int
    ProductCategory_tv: int
    ProductCategory_utility_bill: int

    ChannelId_ChannelId_1: int
    ChannelId_ChannelId_2: int
    ChannelId_ChannelId_3: int
    ChannelId_ChannelId_5: int

class RiskResponse(BaseModel):
    risk_probability: float
