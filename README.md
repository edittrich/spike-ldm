# Prompt 
https://gemini.google.com/share/90f67a184ee6

    ollama pull qwen2.5-coder:32b
    ollama pull llama3.3:70b
    
    ollama create ldm-expert -f ldm-expert.modelfile
    ollama run ldm-expert

Prompt01
"Generate a logical model for this attributes: order_id, order_date, customer_id, customer_name, customer_tier, product_id, product_name, product_category, quantity, unit_price, line_total, shipping_address_street, shipping_address_city, shipping_zip."

Prompt02
"Generate a logical model for this attributes: lead_id, lead_source, company_name, industry, contact_person, contact_phone, meeting_date, meeting_notes, assigned_sales_rep."

Prompt03
"Generate a logical model for this attributes: org_id, org_name, org_tax_id, contact_person_1_name, contact_person_1_role, contact_person_1_phone, contact_person_2_name, contact_person_2_role, contact_person_2_phone, subscription_plan_id, subscription_start_date, subscription_status, invoice_id, invoice_amount, invoice_due_date, invoice_paid_status, service_ticket_id, service_ticket_priority, service_ticket_desc, assigned_agent_id, assigned_agent_name."

# Train
    uv add --dev ruff
    uv add torch
    uv add accelerate bitsandbytes datasets peft protobuf python-dotenv sentencepiece transformers trl
    
    uv run ruff check --fix
    uv run ruff format
